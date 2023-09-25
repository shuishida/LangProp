from typing import List, Union

import carla
import numpy as np
from carla import Actor

"""
https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""


def get_distance(actor1: Actor, actor2: Actor):
    return actor1.get_location().distance(actor2.get_location())


def get_length(vec: carla.Vector3D):
    return np.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)


def world_to_image(point: Union[carla.Location, List[carla.Location], np.ndarray],
                   intrinsics: np.ndarray, world_to_camera: np.ndarray):
    """
    Calculate 2D projection of 3D coordinate
    https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/

    world_to_camera matrix can be computed by
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    """
    if isinstance(point, (carla.Vector3D, carla.Location)):
        point = np.array([point.x, point.y, point.z])
    elif isinstance(point, list) and isinstance(point[0], (carla.Vector3D, carla.Location)):
        point = np.array([[p.x, p.y, p.z] for p in point])

    if point.shape == (3,):
        point = np.array([*point, 1])
    elif point.shape == (2,):
        point = np.array([*point, 0, 1])
    elif point.shape[1] == 3:
        point = np.concatenate([point, np.ones_like(point[:, 0:1])], axis=1)
    elif point.shape[1] == 2:
        point = np.concatenate([point, np.zeros_like(point[:, 0:1]), np.ones_like(point[:, 0:1])], axis=1)
    elif point.shape[1] == 4 and np.all(point[:, 3] == 1):
        pass
    else:
        raise Exception(f"Points with shape {point.shape} cannot be accepted as world coordinates.")
    # transform to camera coordinates
    front, right, up, _ = np.dot(world_to_camera, point.T)
    # New we must change from UE4's coordinate system to a "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    if np.any(front <= 0):
        raise ValueError

    if front.shape == ():
        point_camera = np.array([right, -up, front])
    else:
        point_camera = np.stack([right, -up, front], axis=0)

    # now project 3D->2D using the camera matrix
    point_img = np.dot(intrinsics, point_camera)
    # normalize
    return (point_img[:2] / point_img[2]).T   # numpy tuple of (width, height)
