#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import os
import time
import traceback
import argparse
from argparse import RawTextHelpFormatter
import sys

from inputimeout import TimeoutOccurred, inputimeout

from data_collect.helpers.route_scenario import DataCollectRouteScenario
from data_collect.helpers.scenario_manager import load_scenario
from data_collect.helpers.statistics_manager import DataCollectStatisticsManager
from data_collect.weather import WEATHERS, WEATHERS_IDS
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.route_indexer import RouteIndexer
from srunner.scenariomanager.carla_data_provider import *

from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError


DENSE_TRAFFIC = bool(os.environ.get("DENSE_TRAFFIC"))


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.depth':      'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.lidar.ray_cast_semantic':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}


class DataCollector(LeaderboardEvaluator):
    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()

            if hasattr(self, 'agent_class') and self.agent_class:
                print("Using loaded agent class...")
            else:
                print("Retrieving agent class...")
                agent_class_name = getattr(self.module_agent, 'get_entry_point')()
                self.agent_class = getattr(self.module_agent, agent_class_name)

            print("Initializing agent...")
            try:
                self.agent_instance = self.agent_class(args.agent_config,
                                                       traffic_manager=self.traffic_manager,
                                                       route_name=config.name,
                                                       repetition=config.repetition_index)
            except:
                self.agent_instance = self.agent_class(args.agent_config)

            if hasattr(self, "brains") and self.brains:
                print("Using saved brain...")
                self.agent_instance.brains = self.brains
            elif hasattr(self.agent_instance, "setup_brains"):
                print("Setting up brain in data collector...")
                self.agent_instance.brains = self.brains = self.agent_instance.setup_brains()

            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()

                # Skipping sensor config check during data collection
                # track = self.agent_instance.track
                # AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)

            if args.weather:
                try:
                    weather_id = int(args.weather)
                    weather = WEATHERS[WEATHERS_IDS[weather_id]]
                    self.world.set_weather(weather)
                except Exception as e:
                    print(f"Couldn't convert weather_id {args.weather} to int. Using default weather.")

            self._prepare_ego_vehicles(config.ego_vehicles, False)

            if DENSE_TRAFFIC:
                config.town_amount = 500

            scenario = DataCollectRouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(scenario.scenario)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            load_scenario(self.manager, scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)

    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        index = self.statistics_manager.get_next_route_index()
        route_indexer._index = index
        while index < len(route_indexer._configs_list):
            # setup
            key, config = route_indexer._configs_list[index]

            # run
            self._load_and_run_scenario(args, config)

            route_indexer.save_state(args.checkpoint)

            try:
                inputimeout(prompt='If you want to quit now press enter.', timeout=1)
                print("Exiting...")
                sys.exit(-1)
            except TimeoutOccurred:
                pass

            if args.just_one:
                break

            index = self.statistics_manager.get_next_route_index()
            route_indexer._index = index

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        DataCollectStatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', type=int, default=None,
                        help='Port to use for the TrafficManager')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--weather',
                        type=str, default=None,
                        help='Name of the weather to be executed',
                        )
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--just-one', type=bool, default=False, help='Run just one episode.')

    arguments = parser.parse_args()

    if arguments.trafficManagerPort is None:
        arguments.trafficManagerPort = arguments.port + 2

    statistics_manager = DataCollectStatisticsManager()

    try:
        leaderboard_evaluator = DataCollector(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
