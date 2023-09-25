#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a statistics manager for the CARLA AD leaderboard
"""

from __future__ import print_function

import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

from leaderboard.utils.checkpoint_tools import create_default_json_msg, fetch_dict, save_dict
from leaderboard.utils.statistics_manager import StatisticsManager, compute_route_length, RouteRecord
from srunner.scenariomanager.traffic_events import TrafficEventType

PENALTY_COLLISION_PEDESTRIAN = float(os.environ.get("PENALTY_COLLISION_PEDESTRIAN", 0.50))
PENALTY_COLLISION_VEHICLE = float(os.environ.get("PENALTY_COLLISION_VEHICLE", 0.60))
PENALTY_COLLISION_STATIC = float(os.environ.get("PENALTY_COLLISION_STATIC", 0.65))
PENALTY_TRAFFIC_LIGHT = float(os.environ.get("PENALTY_TRAFFIC_LIGHT", 0.70))
PENALTY_STOP = float(os.environ.get("PENALTY_STOP", 0.80))


class DataCollectStatisticsManager(StatisticsManager):

    """
    This is the statistics manager for the CARLA leaderboard.
    It gathers data at runtime via the scenario evaluation criteria.
    """
    def __init__(self):
        super().__init__()
        self._count_route_fails = defaultdict(int)
        self._n_trials = 3

    def get_next_route_index(self):
        for index, route_record in enumerate(self._registry_route_records):
            if route_record.status in ("Failed", "Failed - Simulation crashed",
                                       "Failed - Agent crashed", "Failed - Agent couldn't be set up"):
                self._count_route_fails[index] += 1
                if self._count_route_fails[index] >= self._n_trials:
                    continue
                return index
        return len(self._registry_route_records)

    def compute_route_statistics(self, config, duration_time_system=-1, duration_time_game=-1, failure=""):
        """
        Compute the current statistics by evaluating all relevant scenario criteria
        """
        index = config.index

        if not self._registry_route_records or index >= len(self._registry_route_records):
            raise Exception('Critical error with the route registry.')

        # fetch latest record to fill in
        route_record = self._registry_route_records[index]

        target_reached = False
        score_penalty = 1.0
        score_route = 0.0

        route_record.meta['duration_system'] = duration_time_system
        route_record.meta['duration_game'] = duration_time_game
        route_record.meta['route_length'] = compute_route_length(config)

        if self._master_scenario:
            if self._master_scenario.timeout_node.timeout:
                route_record.infractions['route_timeout'].append('Route timeout.')
                failure = "Agent timed out"

            for node in self._master_scenario.get_criteria():
                if node.list_traffic_events:
                    # analyze all traffic events
                    for event in node.list_traffic_events:
                        if event.get_type() == TrafficEventType.COLLISION_STATIC:
                            score_penalty *= PENALTY_COLLISION_STATIC
                            route_record.infractions['collisions_layout'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                            score_penalty *= PENALTY_COLLISION_PEDESTRIAN
                            route_record.infractions['collisions_pedestrian'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                            score_penalty *= PENALTY_COLLISION_VEHICLE
                            route_record.infractions['collisions_vehicle'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                            score_penalty *= (1 - event.get_dict()['percentage'] / 100)
                            route_record.infractions['outside_route_lanes'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                            score_penalty *= PENALTY_TRAFFIC_LIGHT
                            route_record.infractions['red_light'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                            route_record.infractions['route_dev'].append(event.get_message())
                            failure = "Agent deviated from the route"

                        elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                            score_penalty *= PENALTY_STOP
                            route_record.infractions['stop_infraction'].append(event.get_message())

                        elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                            route_record.infractions['vehicle_blocked'].append(event.get_message())
                            failure = "Agent got blocked"

                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                            score_route = 100.0
                            target_reached = True
                        elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                            if not target_reached:
                                if event.get_dict():
                                    score_route = event.get_dict()['route_completed']
                                else:
                                    score_route = 0

        # update route scores
        route_record.scores['score_route'] = score_route
        route_record.scores['score_penalty'] = score_penalty
        route_record.scores['score_composed'] = max(score_route*score_penalty, 0.0)

        # update status
        if target_reached:
            route_record.status = 'Completed'
        else:
            route_record.status = 'Failed'
            if failure:
                route_record.status += ' - ' + failure

        return route_record

    @staticmethod
    def save_record(route_record, index, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()

        stats_dict = route_record.__dict__
        record_list = data['_checkpoint']['records']
        if index > len(record_list):
            print('Error! No enough entries in the list')
            sys.exit(-1)
        elif index == len(record_list):
            record_list.append(stats_dict)
        else:
            record_list[index] = stats_dict

        save_dict(endpoint, data)

        now = datetime.now()
        save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

        save_dict(endpoint + "." + save_name, data)
