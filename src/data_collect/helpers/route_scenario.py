#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import carla

from leaderboard.scenarios.route_scenario import RouteScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class DataCollectRouteScenario(RouteScenario):
    town_amount = {
        'Town01': 120,
        'Town02': 100,
        'Town03': 120,
        'Town04': 200,
        'Town05': 120,
        'Town06': 150,
        'Town07': 110,
        'Town08': 180,
        'Town09': 300,
        'Town10': 120,
    }

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Create the background activity of the route
        if hasattr(config, "town_amount"):
            amount = config.town_amount
        else:
            amount = self.town_amount[config.town] if config.town in self.town_amount else 0
        print(f"Setting town traffic density to {amount}")

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)
