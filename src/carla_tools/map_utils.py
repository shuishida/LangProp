import math
import numpy as np
import carla
from carla import World, Location
import xml.etree.ElementTree as ET


EARTH_RADIUS_EQUA = 6378137.0  # pylint: disable=invalid-name


def get_latlon_ref(world_map: carla.Map=None):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    # default reference
    lat_ref = 0.0
    lon_ref = 0.0

    if world_map is not None:
        xodr = world_map.to_opendrive()
        tree = ET.ElementTree(ET.fromstring(xodr))

        for opendrive in tree.iter("OpenDRIVE"):
            for header in opendrive.iter("header"):
                for georef in header.iter("geoReference"):
                    if georef.text:
                        str_list = georef.text.split(' ')
                        for item in str_list:
                            if '+lat_0' in item:
                                lat_ref = float(item.split('=')[1])
                            if '+lon_0' in item:
                                lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def location_to_gps(location: Location, world_map: carla.Map = None):
    """
    https://github.com/carla-simulator/leaderboard/blob/master/leaderboard/utils/route_manipulation.py

    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """
    lat_ref, lon_ref = get_latlon_ref(world_map)

    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}


def gps_to_location(lat: float, lon: float, z: float = 0.0, world_map: carla.Map = None):
    """
    Inverse operation of location_to_gps

    Convert from GPS coordinates to world coordinates
    """
    lat_ref, lon_ref = get_latlon_ref(world_map)

    # Calculate radius of gps point to the earth's vertical axis
    scale = math.cos(lat_ref * math.pi / 180.0)
    radius = scale * EARTH_RADIUS_EQUA

    center_x = radius * lon_ref * math.pi / 180.0
    center_y = radius * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))

    x_coords = lon * (math.pi * radius) / 180.0
    y_coords = math.log(math.tan((lat + 90.0) / 360.0 * math.pi)) * radius

    rel_loc_x = x_coords - center_x
    rel_loc_y = center_y - y_coords

    return carla.Location(x=rel_loc_x, y=rel_loc_y, z=z)


def gps_to_pos(lat: float, lon: float, z: float = 0.0, mean=(0.0, 0.0), scale=(111324.60662786, 111319.490945)):
    """
    Approximation of gps_to_location for carla 9.10 where (lat_ref, lon_ref) = (0, 0)
    Latitude faces north and longitude faces east, but in Carla x faces east and y faces south
    """
    pos = np.array([lat, lon])
    pos -= np.array(mean)
    north, east = pos * np.array(scale)
    return np.array([east, -north])
