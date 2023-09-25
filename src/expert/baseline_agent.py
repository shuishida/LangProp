import importlib
import os
import pathlib
import sys
from typing import Dict

import yaml

from data_agent.core_agent import AgentBrain
from data_agent.data_agent import DataAgent

from expert.expert_agent import ExpertBrain

SAVE_PATH = os.environ.get('SAVE_PATH', None)
TEAM_CODE_ROOT = pathlib.Path(__file__).parent.parent


def to_teamcode_path(path):
    if not os.path.isabs(path):
        path = TEAM_CODE_ROOT / path
    return path


def get_entry_point():
    return 'BaselineAgent'


class BaselineBrain(AgentBrain):
    def __init__(self, config, global_plan_gps, global_plan_world_coord):
        self.config = config
        brain_config = config["brain"]["baseline"]
        agent_path = to_teamcode_path(brain_config["agent"])
        agent_config = to_teamcode_path(brain_config["config"])

        module_name = os.path.basename(agent_path).split('.')[0]
        sys.path.insert(0, os.path.dirname(agent_path))
        driving_model_module = importlib.import_module(module_name)

        agent_class_name = getattr(driving_model_module, 'get_entry_point')()
        agent_class = getattr(driving_model_module, agent_class_name)

        self.agent_model = agent_class(agent_config)
        self.agent_model.set_global_plan(global_plan_gps, global_plan_world_coord)

    def get_control(self, input_data, timestamp, tick_data, step: int) -> dict:
        control = self.agent_model.run_step(self.preprocess(input_data, timestamp), timestamp)
        return dict(
            steer=control.steer,
            throttle=control.throttle,
            brake=control.brake,
        )

    def preprocess(self, input_data, timestamp):
        input_data = {k: (timestamp, v) for k, v in input_data.items()}
        for key_before, key_after in self.config["sensor"]["map"].items():
            input_data[key_after] = input_data.pop(key_before)
        return input_data


class BaselineAgent(DataAgent):
    def setup_brains(self) -> Dict[str, AgentBrain]:
        return {
            "expert": ExpertBrain(self.config),
            "baseline": BaselineBrain(self.config, self._global_plan_dense, self._global_coords_dense),
        }
