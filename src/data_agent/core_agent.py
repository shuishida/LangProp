import os
import datetime
import pathlib
from queue import Empty
from typing import Dict

import yaml

import carla
from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from srunner.scenariomanager.timer import GameTime

SAVE_PATH = os.environ.get("SAVE_PATH", None)
ROUTES = os.environ["ROUTES"]
REPETITIONS = int(os.environ["REPETITIONS"])


def get_entry_point():
    return "CoreAgent"


class AgentBrain:
    def reset(self, agent: 'CoreAgent'):
        pass

    def get_control(self, input_data: dict, timestamp, tick_data: dict, step: int) -> dict:
        raise NotImplementedError


class CoreAgent(autonomous_agent.AutonomousAgent):
    def __init__(self, path_to_conf_file, traffic_manager=None, route_name=None, repetition=0, **kwargs):
        super().__init__(path_to_conf_file)
        self.traffic_manager = traffic_manager
        self.save_path = self._get_save_path(route_name, repetition)
        if self.save_path:
            print(f"Save path: {self.save_path}")

        self.step = 0
        self.initialized = False
        GameTime._last_frame = 0

        self.sensors_data = {}

        self._control = None
        self.brains = None

    def _get_save_path(self, route_name, repetition):
        if not SAVE_PATH:
            return None
        save_dir = pathlib.Path(ROUTES).stem + "_"
        now = datetime.datetime.now()
        save_dir += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

        if route_name:
            save_dir += "_" + route_name
            if REPETITIONS > 1:
                save_dir += f"_{repetition}"

        return pathlib.Path(SAVE_PATH) / save_dir

    def __call__(self):
        if not self.initialized:
            self._init()
            self.initialized = True
            self.sensors_data = {}
            self.start_game_frame = GameTime.get_frame()
            if self.brains is None:
                print("Initializing brain...")
                self.brains = self.setup_brains()
            print("Resetting brain...")
            for brain in self.brains.values():
                brain.reset(self)

        curr_game_frame = GameTime.get_frame()

        if curr_game_frame >= self.start_game_frame + self.eval_freq * self.step:
            game_frames = {}

            try:
                while len(game_frames.keys()) < len(self.sensor_interface._sensors_objects.keys()):

                    # Don't wait for the opendrive sensor
                    if self.sensor_interface._opendrive_tag and self.sensor_interface._opendrive_tag not in game_frames.keys() \
                            and len(self.sensor_interface._sensors_objects.keys()) == len(game_frames.keys()) + 1:
                        break

                    sensor_type, frame_id, sensor_data = self.sensor_interface._new_data_buffers.get(True, timeout=1)
                    self.sensors_data[sensor_type] = sensor_data
                    game_frames[sensor_type] = frame_id

            except Empty:
                pass

            timestamp = GameTime.get_time()

            if self.step % self.save_freq == 0:
                # Debug game frame
                print(f"Current game frame: {curr_game_frame}", game_frames, end='\r')

            self._control = self.run_step(self.sensors_data, timestamp)
            self._control.manual_gear_shift = False

            self.step += 1

        return self._control

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS

        with open(path_to_conf_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.eval_freq = self.config.get("eval_freq", 1)
        self.save_freq = self.config.get("save_freq", 1)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._global_plan_dense = global_plan_gps
        self._global_coords_dense = global_plan_world_coord

    def run_step(self, input_data, timestamp):
        tick_data = self.process(input_data, timestamp)
        tick_data["control"] = {}

        control_info = None
        for name, brain in self.brains.items():
            tick_data["control"][name] = brain.get_control(input_data=input_data, timestamp=timestamp,
                                                           tick_data=tick_data, step=self.step)
            if control_info is None or name == self.config.get("brain", {}).get("main"):
                control_info = tick_data["control"][name]

        tick_data["control"]["main"] = control_info
        control = carla.VehicleControl()
        control.steer = control_info["steer"]
        control.throttle = control_info["throttle"]
        control.brake = control_info["brake"]

        if self.save_path is not None and (self.step % self.save_freq) == 0:
            self.save(tick_data)

        return control

    def _init(self):
        """
        Called at the start of run_step()
        """
        raise NotImplementedError

    def setup_brains(self) -> Dict[str, AgentBrain]:
        raise NotImplementedError

    def sensors(self):
        raise NotImplementedError

    def process(self, input_data, timestamp):
        raise NotImplementedError

    def save(self, tick_data):
        raise NotImplementedError
