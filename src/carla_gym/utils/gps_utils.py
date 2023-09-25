"""Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license."""

import math
import carla
import numpy as np
from carla_gym.core.task_actor.common.navigation.route_manipulation import gps_to_location
from carla_gym.utils.transforms import vec_global_to_ref

EARTH_RADIUS_EQUA = 6378137.0


def gps2xyz(lat, lon, z, lat_ref=49.0, lon_ref=8.0):
    # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)

    mx = lon / 180.0 * (math.pi * EARTH_RADIUS_EQUA * scale)
    my = math.log(math.tan((lat+90.0)*math.pi/360.0))*(EARTH_RADIUS_EQUA * scale)

    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my

    return x, y, z


def xyz2gps(x, y, z, lat_ref=49.0, lon_ref=8.0):
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += x
    my -= y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    return lat, lon, z


def preprocess_gps(ego_gps, target_gps, imu):
    # imu nan bug
    compass = 0.0 if np.isnan(imu[-1]) else imu[-1]
    target_vec_in_global = gps_to_location(target_gps) - gps_to_location(ego_gps)
    ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
    loc_in_ev = vec_global_to_ref(target_vec_in_global, ref_rot_in_global)
    return loc_in_ev
