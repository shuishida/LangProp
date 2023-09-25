from typing import Dict

from data_agent.core_agent import AgentBrain
from data_agent.data_agent import DataAgent

from expert.detect_hazards import calc_dist_to_hazards
from expert.geometry_utils import get_heading_angle
from carla_tools.pid_controller import PIDController

import numpy as np


def get_entry_point():
    return "ExpertAgent"


class ExpertBrain(AgentBrain):
    def __init__(self, config):
        self.config = config
        self.brain_config = config["brain"]["expert"]
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self.completed_stop_sign = {}

    def reset(self, agent):
        self._turn_controller.reset()
        self._speed_controller.reset()
        self.completed_stop_sign = {}

    def get_control(self, tick_data, step: int, **kwargs):
        speed = tick_data["measurements"]["speed"]
        angle, query_angle = (get_heading_angle(tick_data["target"][k]) for k in ("near_vec", "query_vec"))

        # Steering.
        steer = self._turn_controller.step(angle / 90.0)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        target_speed = 6.0

        vis_hazards = self.calc_dist_to_hazards(tick_data, step, angle,
                                                self.brain_config["time_margin"],
                                                self.brain_config["dist_margin"] + 2, actor_margin=0.25)

        should_slow_hazards = self.calc_dist_to_hazards(tick_data, step, angle, self.brain_config["time_margin"], self.brain_config["dist_margin"], actor_margin=0)

        should_slow = sum(len(v) for v in should_slow_hazards.values()) > 0

        should_stop_hazards = self.calc_dist_to_hazards(tick_data, step, angle, time_margin=0, dist_margin=self.brain_config["dist_margin"], actor_margin=0)

        should_stop = sum(len(v) for v in should_stop_hazards.values()) > 0

        if should_stop:
            target_speed = 0
            speed_level = "STOP"
        elif should_slow:
            target_speed = self._get_target_speed(should_slow_hazards, min_speed=1.0, max_speed=target_speed)
            speed_level = "SLOW"
        else:
            speed_level = "MOVE"

        delta = target_speed - speed

        accel = self._speed_controller.step(delta)
        throttle = np.clip(accel, 0.0, 0.75)
        brake = np.clip(-accel, 0.0, 1.0)
        if should_stop:
            throttle = 0.0
            brake = 1.0

        return dict(
            steer=steer,
            throttle=throttle,
            brake=brake,
            angle=angle,
            should_slow=should_slow,
            should_stop=should_stop,
            speed_level=speed_level,
            query_angle=query_angle,
            target_speed=target_speed,
            hazards=vis_hazards,
        )

    def calc_dist_to_hazards(self, tick_data, step, angle, time_margin, dist_margin, **kwargs):
        dist_to_hazards = calc_dist_to_hazards(tick_data["ego_info"], tick_data["actors_info"],
                                               tick_data["target"]["near_command"], angle,
                                               time_margin=time_margin, dist_margin=dist_margin, **kwargs)
        incomplete_stop_signs = {}
        for stop_sign_id, dist in dist_to_hazards["stop_sign"].items():
            stop_completed = tick_data["ego_info"]["speed"] < self.brain_config["stop_sign"]["speed_thresh"]
            if stop_completed:
                self.completed_stop_sign[stop_sign_id] = step
            else:
                stop_effective_max_step = self.completed_stop_sign.get(stop_sign_id, -np.inf) + self.brain_config["stop_sign"]["valid_steps"]
                if step > stop_effective_max_step:
                    incomplete_stop_signs[stop_sign_id] = dist

        dist_to_hazards["stop_sign"] = incomplete_stop_signs

        return dist_to_hazards

    def _get_target_speed(self, hazards: Dict[str, Dict[int, float]], min_speed, max_speed):
        target_speed_list = [max_speed]
        for hazard_type, hazard in hazards.items():
            for hazard_id, dist in hazard.items():
                if hazard_type == "red_light":
                    target_speed = dist / self.brain_config["time_margin"]
                elif hazard_type == "stop_sign":
                    target_speed = dist / self.brain_config["time_margin"]
                else:
                    target_speed = (dist - self.brain_config["dist_margin"]) / self.brain_config["time_margin"]
                target_speed_list.append(target_speed)
        return max(min_speed, min(target_speed_list))


class ExpertAgent(DataAgent):
    def setup_brains(self) -> Dict[str, AgentBrain]:
        return {
            "expert": ExpertBrain(self.config)
        }
