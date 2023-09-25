import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, OrderedDict

from carla_tools.pid_controller import PIDController

import numpy as np

from data_agent.core_agent import AgentBrain
from data_agent.data_agent import DataAgent

from expert.expert_agent import ExpertBrain
from langprop.module import LPModule, RunConfig
from langprop.utils import set_timeout
from lmdrive.dataset import DrivingLMDataset, get_batch_samples
from lmdrive.trainer import preprocess_data, LMDriveTrainer

DATA_ROOT_BASE = Path(os.environ["DATA_ROOT_BASE"])
RUN_NAME = os.environ.get("RUN_NAME")
SAVE_PATH = Path(os.environ["SAVE_PATH"])
ONLINE_ONLY = os.environ.get("ONLINE_ONLY", False)
IGNORE_INFRACTIONS = os.environ.get("IGNORE_INFRACTIONS", False)
POLICY_CKPT = Path(os.environ["POLICY_CKPT"]) if "POLICY_CKPT" in os.environ else None


def get_entry_point():
    return "LMAgent"


infraction_description = {
    "COLLISION_STATIC": "This means that the ego vehicle collided into a static object.",
    "COLLISION_VEHICLE": "This means that the ego vehicle collided with another vehicle.",
    "COLLISION_PEDESTRIAN": "This means that the ego vehicle collided with a pedestrian.",
    "ROUTE_DEVIATION": "This means that the ego vehicle failed to track the given route and deviated from the route.",
    "ROUTE_COMPLETION": "This means that the ego vehicle failed to complete the route.",
    "TRAFFIC_LIGHT_INFRACTION": "This means that the ego vehicle ignored a red traffic light.",
    "WRONG_WAY_INFRACTION": "This means that the ego vehicle went into a wrong lane.",
    "ON_SIDEWALK_INFRACTION": "This means that the ego vehicle went into a sidewalk.",
    "STOP_INFRACTION": "This means that the ego vehicle ignored a slow sign and went at normal speed.",
    "OUTSIDE_LANE_INFRACTION": "This means that the ego vehicle went outside the lane.",
    "OUTSIDE_ROUTE_LANES_INFRACTION": "This means that the ego vehicle didn't follow the lane that it is supposed to follow.",
    "VEHICLE_BLOCKED": "This means that the ego vehicle was stopping / moving slowly for too long without moving.",
}


class LangPropBrain(AgentBrain):
    def __init__(self, config, save_path):
        self.config = config
        self.brain_config = config["brain"]["langprop"]

        if POLICY_CKPT:
            print(f"Loading policy checkpoint at {POLICY_CKPT}")
            self.policy = LPModule.from_checkpoint(POLICY_CKPT)
        else:
            print("$POLICY_CKPT isn't set. Initialising policy from template.")
            self.policy = LPModule.from_template(name="predict_speed_and_steering",
                                                 root=Path(__file__).parent / "models")

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self.infraction_lookahead = self.config["infraction_lookahead"]

        self.run_config = RunConfig(run_name=RUN_NAME, root_dir=save_path, **self.brain_config["run_config"],
                                      save_config=self.config)
        self.policy.setup(self.run_config)
        self.trainer = LMDriveTrainer(self.policy, self.run_config)

        self.train_dataset = DrivingLMDataset(DATA_ROOT_BASE / self.brain_config["train_path"],
                                              load_jpg=(), load_npy=()) if "train_path" in self.brain_config and not ONLINE_ONLY else []
        self.val_dataset = DrivingLMDataset(DATA_ROOT_BASE / self.brain_config["val_path"],
                                            load_jpg=(), load_npy=()) if "val_path" in self.brain_config else []
        self.replay_dataset = DrivingLMDataset(SAVE_PATH, load_jpg=(), load_npy=(), infraction_lookahead=self.infraction_lookahead)

        self.batch_update_freq = self.brain_config["batch_update_freq"]
        self.replay_batch_size = self.brain_config["replay_batch_size"]
        self.train_batch_size = self.brain_config["train_batch_size"]
        self.val_batch_size = self.brain_config["val_batch_size"]

        print("Training samples: ", len(self.train_dataset))
        print("Validation samples: ", len(self.val_dataset))

        self.replay_buffer: List[dict] = list(iter(self.replay_dataset))  # (tick_data, route, index, is_infraction)

        print("Replay buffer size: ", len(self.replay_buffer))

        self.prev_output = None

    def reset(self, agent):
        self._turn_controller.reset()
        self._speed_controller.reset()

        self.run_config.set_dirs(agent.save_path / "lm_policy")

        self.save_path = agent.save_path
        self.prev_output = None

    def get_batch_samples(self):
        return get_batch_samples(self.train_dataset, self.replay_buffer, self.config["infraction_weight"],
                                 self.train_batch_size, self.replay_batch_size)

    def train_on_buffer(self, step):
        print(f"Updating policy using replay buffer. Current time: {datetime.now()}")
        tag = f"{step:06d}_batch_update"
        batch = self.get_batch_samples()
        if batch:
            time_info = self.trainer.fit_batch(batch, tag, step)
            print(time_info)
        else:
            print("Batch is empty. Skipping training on buffer.")

    def update_upon_exception(self, step, buffer_items, attempts=None):
        if attempts is None:
            print(f"Updating policy upon infraction at step {step}. Current time: {datetime.now()}")
        else:
            print(f"Updating policy upon exception at step {step}. Attempt: {attempts}. Current time: {datetime.now()}")

        tag = f"{step:06d}_exception"
        if attempts is not None:
            tag += f"_{attempts}"

        batch = self.get_batch_samples() + buffer_items * self.config["infraction_weight"]
        time_info = self.trainer.fit_batch(batch, tag, step, sort_pre_update=False)
        print(time_info)

    def val(self, step):
        sampled_val_data = list(np.random.choice(self.val_dataset, self.val_batch_size))
        self.trainer.val_batch(sampled_val_data, step)

    def exec_policy_until_success(self, tick_data, step: int, attempts: int = 0):
        inputs, label = preprocess_data(tick_data)

        try:
            with set_timeout(self.run_config.forward_timeout):
                output = self.policy(inputs)
                self.trainer.test_output(output, (inputs,), {}, label)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            if attempts < self.brain_config["n_attempts"]:
                self.update_upon_exception(step, [tick_data], attempts=attempts)
                return self.exec_policy_until_success(tick_data, step, attempts=attempts + 1)
            else:
                return self.prev_output

        return output

    def get_control(self, tick_data, step: int, **kwargs):
        _index = step // self.config["save_freq"]
        tick_data["infractions_ahead"] = []
        if IGNORE_INFRACTIONS:
            tick_data["infractions"] = []

        infractious_buffer_items = []
        if step % self.config["save_freq"] == 0 and tick_data["infractions"]:
            for i in range(max(0, len(self.replay_buffer) - self.infraction_lookahead), len(self.replay_buffer)):
                buffer_item = self.replay_buffer[i]
                if buffer_item["data_dir"] == self.save_path and _index - self.infraction_lookahead <= buffer_item["index"]:
                    buffer_item["infractions_ahead"].extend(tick_data["infractions"])
                    infractious_buffer_items.append(buffer_item)

            if infractious_buffer_items:
                self.update_upon_exception(step, infractious_buffer_items)

        if step % (self.config["save_freq"] * self.batch_update_freq) == 0:
            if not infractious_buffer_items:
                self.train_on_buffer(step)
            self.val(step)

        self.prev_output = output = self.exec_policy_until_success(tick_data, step)

        speed_level, angle = output

        # Steering.
        steer = self._turn_controller.step(angle / 90.0)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        speed = tick_data["measurements"]["speed"]

        target_speed = 6.0

        if speed_level == "STOP":
            target_speed = 0
        elif speed_level == "SLOW":
            target_speed = 0.1

        delta = target_speed - speed

        accel = self._speed_controller.step(delta)
        throttle = 0 if speed_level == "STOP" else np.clip(accel, 0.0, 0.75)
        brake = 1 if speed_level == "STOP" else np.clip(-accel, 0.0, 1.0)

        if step % self.config["save_freq"] == 0:
            buffer_item = {**tick_data, "data_dir": self.save_path, "index": _index}
            buffer_item["control"]["main"] = {"speed_level": speed_level}
            self.replay_buffer.append(buffer_item)

        return dict(
            steer=steer,
            throttle=throttle,
            brake=brake,
            speed_level=speed_level,
            delta=delta,
            accel=accel,
            target_speed=target_speed,
            angle=angle
        )


class LMAgent(DataAgent):
    def setup_brains(self) -> Dict[str, AgentBrain]:
        return {
            "expert": ExpertBrain(self.config),
            # "baseline": BaselineBrain(self.config, self._global_plan_dense, self._global_coords_dense),
            "langprop": LangPropBrain(self.config, self.save_path / "lm_policy")
        }
