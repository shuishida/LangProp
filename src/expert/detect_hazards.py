from typing import Dict, List, Optional, Callable
import numpy as np
from shapely.geometry import Polygon

from agents.navigation.local_planner import RoadOption
from data_agent.process_data import AgentInfo, ActorInfo, BBox3dDict
from expert.geometry_utils import normalize_deg
from carla_tools.transforms import rotate_in_plane


def calc_dist_to_hazards(ego_info: AgentInfo, actors_info: Dict[str, Dict[int, ActorInfo]], command, steer_angle, time_margin, dist_margin, **kwargs):
    if command == RoadOption.CHANGELANELEFT:
        lane_vehicles = margin_to_actors(ego_info, actors_info["vehicle"], steer_angle, time_margin, dist_margin, ego_left_margin=2.0, fov=360)
    elif command == RoadOption.CHANGELANERIGHT:
        lane_vehicles = margin_to_actors(ego_info, actors_info["vehicle"], steer_angle, time_margin, dist_margin, ego_right_margin=2.0, fov=360)
    else:
        lane_vehicles = {}

    dist_thresh = time_margin * ego_info["speed"] + dist_margin
    stop_dist = ego_info["stop_sign_distance"]
    light_dist = ego_info["red_light_distance"]

    results = dict(
        vehicle=margin_to_actors(ego_info, actors_info["vehicle"], steer_angle, time_margin, dist_margin, **kwargs),
        walker=margin_to_actors(ego_info, actors_info["walker"], steer_angle, time_margin, dist_margin, **kwargs),
        stop_sign={ego_info["stop_sign_id"]: stop_dist} if stop_dist is not None and stop_dist < dist_thresh else {},
        red_light={ego_info["traffic_light_id"]: light_dist} if light_dist is not None and light_dist < dist_thresh else {},
        lane_vehicle=lane_vehicles
    )

    return results


def get_polygon(bbox: BBox3dDict, min_forward=0.0, max_forward=0.0, turn_angle=0.0, left_margin=0.0, right_margin=0.0):
    loc = bbox["loc"]
    ori = rotate_in_plane(turn_angle, bbox["ori"])
    right = rotate_in_plane(turn_angle, bbox["right"])
    up = rotate_in_plane(turn_angle, bbox["up"])
    extent = bbox["extent"]
    delta_forward = (extent.x + max_forward) * ori
    delta_behind = (min_forward - extent.x) * ori
    delta_right = (extent.y + right_margin) * right
    delta_left = (- extent.y - left_margin) * right
    delta_top = extent.z * up
    delta_bottom = - extent.z * up

    # Ground projection of the forward-bottom, backward-top rectangle of the actor
    corners = [loc + delta_forward + delta_left + delta_bottom, loc + delta_forward + delta_right + delta_bottom,
               loc + delta_behind + delta_right + delta_top, loc + delta_behind + delta_left + delta_top]
    return Polygon([(corner.x, corner.y) for corner in corners])


def predict_collision(ego_info: AgentInfo, actor_info: ActorInfo, ego_angle, time_margin=0, dist_margin=0, ego_left_margin=0.0, ego_right_margin=0.0, actor_margin=0.25,
                      ego_angle_range=0, actor_angle_range=0, fov=180, interval=0.25):
    ego_speed = ego_info["speed"]

    angle_to_actor = np.degrees(actor_info["angle"]["loc"]) - ego_angle
    angle_to_actor = np.abs(normalize_deg(angle_to_actor))

    if angle_to_actor > fov / 2:
        # Ignore agents out of fov, since we assume actors behind the vehicle won't crash into us
        return None

    prev_time = 0
    for curr_time in np.arange(0, time_margin + interval, interval):
        min_forward = prev_time * ego_speed
        max_forward = curr_time * ego_speed + dist_margin
        ego_side_margin = np.tan(np.radians(ego_angle_range)) * max_forward
        actor_max_forward = curr_time * actor_info["speed"] + actor_margin
        actor_side_margin = np.tan(np.radians(actor_angle_range)) * actor_max_forward + actor_margin
        ego_polygon = get_polygon(ego_info, min_forward=min_forward, max_forward=max_forward, turn_angle=ego_angle,
                                  left_margin=ego_side_margin + ego_left_margin, right_margin=ego_side_margin + ego_right_margin)
        actor_bbox = actor_info["abs"] if actor_info["abs"].get("trigger") is None else actor_info["abs"]["trigger"]
        actor_polygon = get_polygon(actor_bbox, min_forward=0.0, max_forward=actor_max_forward, left_margin=actor_side_margin, right_margin=actor_side_margin)
        if ego_polygon.intersects(actor_polygon):
            return prev_time * ego_speed
        prev_time = curr_time
        if ego_speed == 0:
            break

    return None


def margin_to_actors(ego_info: AgentInfo, actors_data: Dict[int, ActorInfo], ego_angle, time_margin, dist_margin, **kwargs):
    res = {}
    for actor_id, actor_info in actors_data.items():
        dist_collision = predict_collision(ego_info, actor_info, ego_angle, time_margin, dist_margin, **kwargs)
        if dist_collision is not None:
            res[actor_id] = dist_collision
    return res


def margin_to_traffic_light(ego_info: AgentInfo, lights_data: Dict[int, ActorInfo]):
    if ego_info["is_red_light"]:
        light_id = ego_info["traffic_light_id"]
        light_info = lights_data[light_id]
        dist_to_trigger = light_info["rel"]["trigger"]["loc"].x
        return {light_id: dist_to_trigger}
    return {}
