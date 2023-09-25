import os
from pathlib import Path
from typing import Dict

from carla_tools.pid_controller import PIDController

import numpy as np

from data_agent.core_agent import AgentBrain
from data_agent.data_agent import DataAgent
from expert.expert_agent import ExpertBrain

from langprop.module import LPModule, RunConfig
from lmdrive.trainer import preprocess_data

POLICY_CKPT = Path(os.environ["POLICY_CKPT"]) if "POLICY_CKPT" in os.environ else None


def get_entry_point():
    return "LMAgentEval"


class LangPropBrain(AgentBrain):
    def __init__(self, config):
        self.config = config
        self.brain_config = config["brain"]["langprop"]

        if POLICY_CKPT:
            print(f"Loading policy checkpoint at {POLICY_CKPT}")
            self.policy = LPModule.from_checkpoint(POLICY_CKPT)
        else:
            raise Exception("$POLICY_CKPT isn't set. Cannot evaluate langprop agent.")

        self.run_config = RunConfig()
        self.policy.setup(self.run_config)

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def reset(self, agent):
        self._turn_controller.reset()
        self._speed_controller.reset()

    def get_control(self, tick_data, step: int, **kwargs):
        inputs = preprocess_data(tick_data, with_label=False)
        output = self.policy(inputs)

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


class LMAgentEval(DataAgent):
    def setup_brains(self) -> Dict[str, AgentBrain]:
        return {
            "langprop": LangPropBrain(self.config)
        }
