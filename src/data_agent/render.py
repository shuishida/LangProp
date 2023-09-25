from typing import Union, Callable, Dict, List, Optional

import carla
import cv2
import numpy as np
from PIL import Image, ImageDraw

from data_agent.geometry_utils import world_to_image
from data_agent.process_data import BBox
from carla_tools.sensors import CameraConfig


def draw_points_on_image(
        image, world_points: Union[List[carla.Location], np.ndarray], colors: Union[int, float, tuple, list],
        camera_transform: carla.Transform, camera_config: CameraConfig, radius=5,
):
    if len(world_points) == 0:
        return image

    if isinstance(colors, list):
        assert len(world_points) == len(colors)
    else:
        if isinstance(colors, tuple):
            assert len(colors) in (3, 4)
        colors = [colors] * len(world_points)

    world_to_camera = camera_transform.get_inverse_matrix()
    intrinsics = camera_config.intrinsics

    targets = world_to_image(world_points, intrinsics, world_to_camera)

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for (w, h), color in zip(targets, colors):
        if not (0 <= w < camera_config.width and 0 <= h < camera_config.height):
            continue
        draw.ellipse((w - radius, h - radius, w + radius, h + radius), color)

    return np.array(image)


# Remember the edge pairs
BBOX_VERTS_EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def draw_bboxes_on_image(image, actors_bboxes: Dict[str, Dict[str, Dict[str, BBox]]], hazards: Dict[str, List[int]] = None):
    hazard_ids = []
    if hazards:
        for h_ids in hazards.values():
            hazard_ids.extend(h_ids)

    for bbox_type in ("bbox", "trigger"):
        for actor_type, actors in actors_bboxes[bbox_type].items():
            for actor_id, actor_bbox in actors.items():
                verts = np.array(actor_bbox["eight"])

                for node1, node2 in BBOX_VERTS_EDGES:
                    p1 = verts[node1]
                    p2 = verts[node2]

                    R = (actor_id in hazard_ids) * 255
                    G = 0
                    B = (bbox_type == "trigger") * 255
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (R, G, B, 255), 1)

    return image


"""
https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h
"""
CITYSCAPES_PALETTE = dict(
    unlabeled=[0, 0, 0],
    # cityscape colors
    road=[128, 64, 128],
    sidewalk=[244, 35, 232],
    building=[70, 70, 70],
    wall=[102, 102, 156],
    fence=[190, 153, 153],
    pole=[153, 153, 153],
    traffic_light=[250, 170, 30],
    traffic_sign=[220, 220, 0],
    vegetation=[107, 142, 35],
    terrain=[152, 251, 152],
    sky=[70, 130, 180],
    pedestrian=[220, 20, 60],
    rider=[255, 0, 0],
    vehicle=[0, 0, 142],
    truck=[0, 0, 70],
    bus=[0, 60, 100],
    train=[0, 80, 100],
    motorcycle=[0, 0, 230],
    bicycle=[119, 11, 32],
    # carla custom
    static=[110, 190, 160],
    dynamic=[170, 120, 50],
    other=[55, 90, 80],
    water=[45, 60, 150],
    road_line=[157, 234, 50],
    ground=[81, 0, 81],
    bridge=[150, 100, 100],
    rail_track=[230, 150, 140],
    guard_rail=[180, 165, 180],
    # custom
    red_light=[255, 0, 0],
    yellow_light=[255, 255, 0],
    green_light=[0, 255, 0],
    stop_sign=[255, 128, 0],
    waypoints=[0, 255, 255],
    near_pos=[0, 128, 255],
    query_pos=[128, 255, 0],
    ego_waypoint=[255, 64, 0],
)

"""
See carla.CityObjectLabel for the original label set
"""

CITYSCAPES_ORDER = (
    "unlabeled",
    "building",
    "fence",
    "other",
    "pedestrian",
    "pole",
    "road_line",
    "road",
    "sidewalk",
    "vegetation",
    "vehicle",
    "wall",
    "traffic_sign",  # 12
    "sky",
    "ground",
    "bridge",
    "rail_track",
    "guard_rail",
    "traffic_light",  # 18
    "static",
    "dynamic",
    "water",
    "terrain",
    "red_light",
    "yellow_light",
    "green_light",
    "stop_sign",
    "waypoints",
    "near_pos",
    "query_pos",
    "ego_waypoint",
)

CITYSCAPES_ID = {n: i for i, n in enumerate(CITYSCAPES_ORDER)}

SEG_COLOR_MAP = np.array([CITYSCAPES_PALETTE[k] for k in CITYSCAPES_ORDER])


def map_seg_image(seg_image: np.ndarray, color_map=SEG_COLOR_MAP):
    return color_map[seg_image].astype(np.uint8)
