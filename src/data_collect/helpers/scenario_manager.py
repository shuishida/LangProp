#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from data_collect.helpers.agent_wrapper import DataCollectAgentWrapper
from srunner.scenariomanager.timer import GameTime


def load_scenario(scenario_manager, scenario, agent, rep_number):
    """
    Load a new scenario
    """
    GameTime.restart()
    GameTime._carla_time = 0.0
    GameTime._last_frame = 0

    agent = DataCollectAgentWrapper(agent)

    scenario_manager._agent = agent
    scenario_manager.scenario_class = scenario
    scenario_manager.scenario = scenario.scenario
    scenario_manager.scenario_tree = scenario_manager.scenario.scenario_tree
    scenario_manager.ego_vehicles = scenario.ego_vehicles
    scenario_manager.other_actors = scenario.other_actors
    scenario_manager.repetition_number = rep_number

    # To print the scenario tree uncomment the next line
    # py_trees.display.render_dot_tree(self.scenario_tree)

    agent.setup_sensors(scenario.ego_vehicles[0], scenario_manager._debug_mode)
