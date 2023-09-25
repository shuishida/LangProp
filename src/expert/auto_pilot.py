from typing import Dict

from data_agent.core_agent import AgentBrain
from data_agent.data_agent import DataAgent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
    return "AutoPilot"


class AutoPilotBrain(AgentBrain):
    def __init__(self, traffic_manager):
        self.traffic_manager = traffic_manager
        # route = [loc for loc, cmd in self._command_planner.route]
        # self.traffic_manager.set_path(self._vehicle, route)
        self.ego_vehicle = None

    def reset(self, agent):
        self.ego_vehicle = CarlaDataProvider.get_hero_actor()
        self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 0)
        self.traffic_manager.auto_lane_change(self.ego_vehicle, False)
        self.ego_vehicle.set_autopilot(True, CarlaDataProvider._traffic_manager_port)

    def get_control(self, input_data: dict, timestamp, tick_data: dict, step: int) -> dict:
        control = self.ego_vehicle.get_control()
        steer = control.steer
        throttle = control.throttle
        brake = control.brake

        return dict(
            steer=steer,
            throttle=throttle,
            brake=brake,
        )


class AutoPilot(DataAgent):
    def setup_brains(self) -> Dict[str, AgentBrain]:
        return {
            "autopilot": AutoPilotBrain(self.traffic_manager)
        }
