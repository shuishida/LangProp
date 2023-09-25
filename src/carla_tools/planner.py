import numpy as np
from collections import namedtuple

from carla_tools.map_utils import gps_to_pos


Waypoint = namedtuple("Waypoint", "pos cmd")


def to_relative_pos(abs_pos, ego_pos, ego_theta):
    _rotation = np.array([
        [np.cos(ego_theta), -np.sin(ego_theta)],
        [np.sin(ego_theta), np.cos(ego_theta)]
    ])
    return (abs_pos - ego_pos) @ _rotation


class RoutePlanner:
    def __init__(self, global_plan, min_distance, fov=360):
        self.route = [Waypoint(gps_to_pos(**gps), cmd) for gps, cmd in global_plan]
        self.min_distance = min_distance
        self.fov = fov

    def run_step(self, ego_pos, ego_theta):
        index = 0
        for index, (pos, cmd) in enumerate(self.route[:-1]):
            wp_distance = np.linalg.norm(pos - ego_pos)
            rel_vec = to_relative_pos(pos, ego_pos, ego_theta)
            angle = np.degrees(np.arctan2(rel_vec[1], rel_vec[0]))
            if wp_distance > self.min_distance:     # and abs(angle) < self.fov / 2:
                break

        self.route = self.route[index:]
        pos, cmd = self.route[0]
        return pos, to_relative_pos(pos, ego_pos, ego_theta), cmd
