from typing import Dict, Optional, List, Tuple, Union, Iterable

import carla
import numpy as np
from carla import Vehicle, Actor, TrafficLightState

from data_agent.geometry_utils import get_distance, get_length, world_to_image
from carla_tools.sensors import CameraConfig, SensorConfig
from carla_tools.transforms import inv_transform, inv_rotate_vec, rotate_vec, transform_vec
from typing_extensions import TypedDict


class BBox3dDict(TypedDict):
    loc: carla.Location
    ori: carla.Vector3D
    right: carla.Vector3D
    up: carla.Vector3D
    extent: carla.Vector3D


class ActorGeometry(BBox3dDict):
    vel: carla.Vector3D
    trigger: Optional[BBox3dDict]


class AngleDict(TypedDict):
    loc: float
    ori: float


class ActorInfo(TypedDict):
    type: str
    type_id: str
    abs: ActorGeometry
    rel: ActorGeometry
    distance: float
    speed: float
    angle: AngleDict
    is_junction: bool
    road_id: int
    same_road: bool
    lane_id: int
    same_lane: bool
    opposite_lane: bool
    on_right_lane: bool
    on_left_lane: bool
    lane_type: str
    state: Optional[str]
    is_green: Optional[bool]
    traffic_light_id: Optional[int]
    traffic_light_state: Optional[str]
    is_red_light: Optional[bool]


class AgentInfo(ActorGeometry):
    rel_vel: carla.Vector3D
    speed: float
    is_junction: bool
    road_id: int
    lane_id: int
    lane_type: str
    traffic_light_id: Optional[int]
    traffic_light_state: Optional[str]
    is_red_light: bool
    red_light_distance: Optional[float]
    stop_sign_id: Optional[int]
    stop_sign_distance: Optional[float]


def collect_agent_info(agent: Vehicle, light_waypoints: Dict[carla.Waypoint, carla.TrafficLight],
                       stop_signs: Dict[int, Actor], world_map: carla.Map) -> AgentInfo:
    ego_transform = agent.get_transform()

    data: AgentInfo = get_bbox_3d_as_dict(get_bounding_box(agent), ego_transform)

    waypoint = world_map.get_waypoint(agent.get_location())

    data["vel"] = vel = agent.get_velocity()
    data["rel_vel"] = inv_rotate_vec(ego_transform.rotation, vel)
    data["speed"] = get_length(vel)

    data["is_junction"] = waypoint.is_junction
    data["road_id"] = waypoint.road_id
    data["lane_id"] = waypoint.lane_id
    data["lane_type"] = str(waypoint.lane_type)
    traffic_light = get_traffic_light_for_vehicle(agent, light_waypoints, world_map)
    data["traffic_light_id"] = traffic_light.id if traffic_light else None
    light_state = traffic_light.state if traffic_light else None
    data["traffic_light_state"] = str(light_state) if light_state is not None else None
    data["is_red_light"] = is_red_light = light_state is not None and light_state != TrafficLightState.Green

    data["red_light_distance"] = get_bbox_3d_as_dict(traffic_light.trigger_volume,
                                                     traffic_light.get_transform(),
                                                     ego_transform)["loc"].x if is_red_light else None

    data["stop_sign_id"], data["stop_sign_distance"] = get_dist_to_stop_sign(agent, stop_signs, world_map)
    return data


def collect_actors_info(ego_vehicle: Vehicle, nearby_actors: Dict[str, Dict[int, Actor]],
                        light_waypoints: Dict[carla.Waypoint, carla.TrafficLight], world_map: carla.Map) \
        -> Dict[str, Dict[int, ActorInfo]]:
    ego_waypoint = world_map.get_waypoint(ego_vehicle.get_location())
    ego_transform = ego_vehicle.get_transform()

    objects_data = {}

    for actor_type, actors in nearby_actors.items():
        objects_data[actor_type] = {}
        for actor in actors.values():
            data: ActorInfo = {"abs": {}, "rel": {}, "angle": {}}

            actor_transform = actor.get_transform()

            waypoint = world_map.get_waypoint(actor.get_location())

            data["type"] = actor_type
            data["type_id"] = actor.type_id

            bbox = get_bounding_box(actor)

            data["abs"] = {
                **get_bbox_3d_as_dict(bbox, actor_transform),
                "vel": actor.get_velocity(),
                "trigger": get_bbox_3d_as_dict(actor.trigger_volume, actor_transform) if actor_type in ("traffic_light", "stop_sign") else None
            }

            data["rel"] = {
                **get_bbox_3d_as_dict(bbox, actor_transform, ego_transform),
                "vel": inv_rotate_vec(ego_transform.rotation, data["abs"]["vel"]),
                "trigger": get_bbox_3d_as_dict(actor.trigger_volume, actor_transform, ego_transform) if actor_type in ("traffic_light", "stop_sign") else None
            }

            data["speed"] = get_length(data["abs"]["vel"])

            rel_loc = data["rel"]["loc"]
            rel_ori = data["rel"]["ori"]
            # relative angle
            data["angle"]["loc"] = np.arctan2(rel_loc.y, rel_loc.x)
            data["angle"]["ori"] = np.arctan2(rel_ori.y, rel_ori.x)

            # relative distance
            data["distance"] = get_distance(ego_vehicle, actor)

            data["is_junction"] = waypoint.is_junction
            data["road_id"] = waypoint.road_id
            data["same_road"] = waypoint.road_id == ego_waypoint.road_id
            data["lane_id"] = waypoint.lane_id
            data["same_lane"] = waypoint.lane_id == ego_waypoint.lane_id
            data["opposite_lane"] = waypoint.lane_id * ego_waypoint.lane_id < 0
            if ego_waypoint.lane_id > 0:
                data["on_right_lane"] = waypoint.lane_id == ego_waypoint.lane_id + 1
                data["on_left_lane"] = waypoint.lane_id == ego_waypoint.lane_id - 1
            else:
                data["on_right_lane"] = waypoint.lane_id == ego_waypoint.lane_id - 1
                data["on_left_lane"] = waypoint.lane_id == ego_waypoint.lane_id + 1

            data["lane_type"] = str(waypoint.lane_type)

            if actor_type == "traffic_light":
                data["state"] = str(actor.state)
                data["is_green"] = actor.state == TrafficLightState.Green

            if actor_type == "vehicle":
                traffic_light = get_traffic_light_for_vehicle(actor, light_waypoints, world_map)
                data["traffic_light_id"] = traffic_light.id if traffic_light else None
                light_state = traffic_light.state if traffic_light else None
                data["traffic_light_state"] = str(light_state) if light_state is not None else None
                data["is_red_light"] = traffic_light and light_state != TrafficLightState.Green

            objects_data[actor_type][actor.id] = data
    return objects_data


def get_bbox_3d_as_dict(bounding_box, actor_transform: carla.Transform, ego_transform: Optional[carla.Transform] = None) -> BBox3dDict:
    """
    Returns a list of 3d bounding boxes. Provide ego_transform to get relative bbox.
    """
    bbox = dict(
        loc=transform_vec(actor_transform, bounding_box.location),
        ori=rotate_vec(actor_transform.rotation, bounding_box.rotation.get_forward_vector()),
        right=rotate_vec(actor_transform.rotation, bounding_box.rotation.get_right_vector()),
        up=rotate_vec(actor_transform.rotation, bounding_box.rotation.get_up_vector()),
        extent=bounding_box.extent
    )

    if ego_transform:
        bbox = dict(
            loc=inv_transform(ego_transform, bbox["loc"]),
            ori=inv_rotate_vec(ego_transform.rotation, bbox["ori"]),
            right=inv_rotate_vec(ego_transform.rotation, bbox["right"]),
            up=inv_rotate_vec(ego_transform.rotation, bbox["up"]),
            extent=bbox["extent"],
        )

    return bbox


def weather_to_dict(carla_weather):
    weather = {
        "cloudiness": carla_weather.cloudiness,
        "precipitation": carla_weather.precipitation,
        "precipitation_deposits": carla_weather.precipitation_deposits,
        "wind_intensity": carla_weather.wind_intensity,
        "sun_azimuth_angle": carla_weather.sun_azimuth_angle,
        "sun_altitude_angle": carla_weather.sun_altitude_angle,
        "fog_density": carla_weather.fog_density,
        "fog_distance": carla_weather.fog_distance,
        "wetness": carla_weather.wetness,
        "fog_falloff": carla_weather.fog_falloff,
    }
    return weather


def get_nearby_objects(ego_agent: Vehicle, objects: List[Actor], cutoff_distance=50):
    """Find all actors of a certain type that are close to the vehicle
    """
    return {obj.id: obj for obj in objects if get_distance(ego_agent, obj) <= cutoff_distance}


def get_bounding_box(actor: Actor, get_trigger: bool = False, min_extent=0.5):
    """
    Traffic lights and signs don't have bounding boxes (or in v0.9.11 it is the same as the trigger volume).
    Moreover, some actors like motorbikes have a zero width bounding box. This is a fix for these issues.
    """
    if get_trigger:
        return actor.trigger_volume

    if "traffic.traffic_light" == actor.type_id:
        bbox = carla.BoundingBox(carla.Location(0, 0, 1.5), carla.Vector3D(x=0.6, y=0.6, z=1.5))
    # elif "traffic.stop" == actor.type_id:
    #     bbox = carla.BoundingBox(carla.Location(0, -4, 1), carla.Vector3D(x=1.5, y=1.0, z=0.5))
    elif hasattr(actor, "bounding_box"):
        bbox = actor.bounding_box
    else:
        # print(f"Bounding box for type {actor.type_id} not implemented. Returning default.")
        bbox = carla.BoundingBox(carla.Location(0, 0, 1), carla.Vector3D(x=1.0, y=1.0, z=1.0))

    # Fixing bug related to Carla 9.11 onwards where the bounding box location is wrongly registered for some 2-wheeled vehicles
    # https://github.com/carla-simulator/carla/issues/3801
    buggy_bbox = (bbox.extent.x * bbox.extent.y * bbox.extent.z == 0)
    if buggy_bbox:
        # print(f"Buggy bounding box found for {actor}")
        bbox.location = carla.Location(0, 0, max(bbox.extent.z, min_extent))
    # Fixing bug related to Carla 9.11 onwards where some bounding boxes have 0 extent
    # https://github.com/carla-simulator/carla/issues/3670
    if bbox.extent.x < min_extent:
        bbox.extent.x = min_extent
    if bbox.extent.y < min_extent:
        bbox.extent.y = min_extent
    if bbox.extent.z < min_extent:
        bbox.extent.z = min_extent

    return bbox


ImgPnt = Tuple[float, float]


class BBox(TypedDict):
    """
    https://carla.org/Doxygen/html/d2/dfe/LibCarla_2source_2carla_2geom_2BoundingBox_8h_source.html#l00076
    [(-x,-y,-z), (-x,-y, z), (-x, y,-z), (-x, y, z), ( x,-y,-z), ( x,-y, z), ( x, y,-z), ( x, y, z)]
    """
    eight: Union[List[ImgPnt], Tuple[ImgPnt, ImgPnt, ImgPnt, ImgPnt, ImgPnt, ImgPnt, ImgPnt, ImgPnt]]
    four: Union[List[float], Tuple[float, float, float, float]]                         # (w_min, h_min, w_max, h_max)
    abs: BBox3dDict
    rel: BBox3dDict


def get_image_bboxes(actors: Iterable[Actor], depth_img, camera_transform: carla.Transform, camera_config: CameraConfig,
                     thresh=1.0, get_trigger: bool = False) -> Dict[int, BBox]:
    """
    Returns a dict of 3d and 2d bounding boxes given a list of actors and the camera transform
    """
    world_to_camera = camera_transform.get_inverse_matrix()
    intrinsics = camera_config.intrinsics

    bboxes: Dict[int, BBox] = {}

    for actor in actors:
        actor_rel_pos = inv_transform(camera_transform, actor.get_location())

        if actor_rel_pos.x <= 0:     # check that the object is in front of the camera
            continue

        actor_transform = actor.get_transform()
        actor_bbox = get_bounding_box(actor, get_trigger)
        world_points = actor_bbox.get_world_vertices(actor_transform)  # locations of 8 corners of the cuboid
        try:
            image_points = world_to_image(world_points, intrinsics, world_to_camera)    # (8, 2)
        except ValueError:
            continue

        w_min = image_points[:, 0].min()
        w_max = image_points[:, 0].max()
        h_min = image_points[:, 1].min()
        h_max = image_points[:, 1].max()

        if w_max - w_min < 1 or h_max - h_min < 1:
            continue

        if w_max > 0 and w_min < camera_config.width and h_max > 0 and h_min < camera_config.height:
            actor_distance = actor.get_location().distance(camera_transform.location)
            depth_in_box = depth_img[int(max(h_min, 0)):int(h_max), int(max(w_min, 0)):int(w_max)]
            if not depth_in_box.size:
                continue

            if actor_distance > depth_in_box.max() + thresh:
                # actor is not directly visible from the agent
                continue

            bboxes[actor.id] = dict(
                eight=image_points.tolist(),
                four=(w_min, h_min, w_max, h_max),
                abs=get_bbox_3d_as_dict(actor_bbox, actor_transform),
                rel=get_bbox_3d_as_dict(actor_bbox, actor_transform, camera_transform)
            )

    return bboxes


def get_bbox_mask(bbox: Tuple[float, float, float, float], image_shape):
    height, width = image_shape[:2]
    w_ind, h_ind = np.meshgrid(np.arange(width), np.arange(height))
    w_min, h_min, w_max, h_max = bbox
    return (w_min <= w_ind) & (w_ind < w_max) & (h_min <= h_ind) & (h_ind < h_max)


def convert_depth(data):
    """Transforms the depth image into meters

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    normalized = np.sum(data * np.array([65536, 256, 1]), axis=-1).astype(np.float32)
    normalized /= 256 * 256 * 256 - 1
    in_meters = 1000 * normalized

    return in_meters


def splat_points(point_cloud, range_in_meters=32, pixels_per_meter=4, hist_max_per_pixel=5):
    """
    Due to the lidar rotated by yaw of -90, x corresponds to right (small) - left (large), and y corresponds to forward.
    """
    # 128 x 128 grid by default
    xbins = np.linspace(- range_in_meters / 2, range_in_meters / 2, range_in_meters * pixels_per_meter + 1)
    ybins = np.linspace(- range_in_meters, 0, range_in_meters * pixels_per_meter + 1)
    # negating the coordinates so that x becomes left-right, and y becomes far-close. Easier to visualise.
    hist = np.histogramdd(-point_cloud[..., :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    overhead_splat = hist / hist_max_per_pixel
    return overhead_splat.T     # from (left-right, far-close) to image space


def lidar_to_histogram_features(lidar, lidar_config: SensorConfig, eps=0.2):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    above = lidar[lidar[..., 2] > -lidar_config.z + eps]
    below = lidar[lidar[..., 2] <= -lidar_config.z + eps]
    above_features = splat_points(above)
    below_features = splat_points(below)
    features = np.concatenate([above_features, below_features], axis=0)
    return features


def get_traffic_light_triggers(traffic_light: carla.TrafficLight) -> Tuple[List[carla.Location], carla.Location]:
    base_transform = traffic_light.get_transform()
    trigger_center = traffic_light.trigger_volume.location
    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent

    trigger_points = []
    for x in np.linspace(- area_ext.x, area_ext.x, int(2 * area_ext.x) + 1):  # check for x every meter
        point = transform_vec(base_transform, trigger_center + carla.Vector3D(x))
        trigger_points.append(carla.Location(point))
    return trigger_points, transform_vec(base_transform, trigger_center)


def follow_waypoint_to_intersect(waypoint: carla.Waypoint):
    while not waypoint.is_intersection:
        wp_nexts = waypoint.next(0.5)
        if not wp_nexts:
            break
        waypoint = wp_nexts[0]
    return waypoint


def get_traffic_light_waypoints(traffic_lights: Dict[int, carla.TrafficLight], world_map: carla.Map):
    traffic_light_waypoints = {}

    for light_id, traffic_light in traffic_lights.items():
        trigger_points, trigger_center = get_traffic_light_triggers(traffic_light)

        center_wp = world_map.get_waypoint(trigger_center)

        # Get the waypoints of these points, removing duplicates
        init_wps = []
        for point in trigger_points:
            waypoint = world_map.get_waypoint(point)
            if waypoint.lane_id * center_wp.lane_id < 0:    # check if the lane is in the same direction
                continue
            # As x_values are arranged in order, only the last one has to be checked
            if not init_wps or init_wps[-1].road_id != waypoint.road_id or init_wps[-1].lane_id != waypoint.lane_id:
                init_wps.append(waypoint)

        # Advance them until the intersection
        for waypoint in init_wps:
            waypoint = follow_waypoint_to_intersect(waypoint)
            traffic_light_waypoints[waypoint] = traffic_light

    return traffic_light_waypoints


def get_traffic_light_for_vehicle(vehicle: carla.Vehicle, light_waypoints: Dict[carla.Waypoint, carla.TrafficLight],
                                  world_map: carla.Map, margin_before=50, margin_after=0.5) -> Optional[carla.TrafficLight]:
    """
    vehicle.get_traffic_light() has a weird behaviour that it does not necessarily return the
    immediate traffic light in front.
    Here, we solve the problem in by using waypoints.
    """
    loc = vehicle.get_location()
    rotation = vehicle.get_transform().rotation

    loc_wp = world_map.get_waypoint(loc)
    loc_wp = follow_waypoint_to_intersect(loc_wp)

    closest_light = None
    min_dist_so_far = 3.0
    for light_wp, light in light_waypoints.items():
        if light_wp.lane_id * loc_wp.lane_id < 0:
            continue
        dist_to_light = inv_rotate_vec(rotation, light_wp.transform.location - loc).x
        if dist_to_light > margin_before or dist_to_light < - margin_after:
            continue
        dis = loc_wp.transform.location.distance(light_wp.transform.location)
        if dis <= min_dist_so_far:
            min_dist_so_far = dis
            closest_light = light
    return closest_light


def is_agent_affected_by_stop(agent: Actor, stop: Actor, world_map: carla.Map, multi_step=20):
    """
    Check if the given actor is affected by the stop
    """
    # first we run a fast coarse test
    if get_distance(agent, stop) > 30.0:
        return False

    # slower and accurate test based on waypoint's horizon and geometric test
    agent_location = agent.get_location()
    waypoint_locations = [agent_location]
    waypoint = world_map.get_waypoint(agent_location)
    for _ in range(multi_step):
        waypoint = waypoint.next(1.0)[0]
        if not waypoint:
            break
        waypoint_locations.append(waypoint.transform.location)

    for location in waypoint_locations:
        if stop.trigger_volume.contains(location, stop.get_transform()):
            return True

    return False


def get_dist_to_stop_sign(agent: Actor, stop_signs: Dict[str, Actor], world_map: carla.Map) -> Tuple[Optional[str], Optional[float]]:
    ego_transform = agent.get_transform()
    ve_dir = ego_transform.get_forward_vector()
    wp = world_map.get_waypoint(ego_transform.location)
    wp_dir = wp.transform.get_forward_vector()

    dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

    if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
        for stop_id, stop in stop_signs.items():
            if is_agent_affected_by_stop(agent, stop, world_map):
                # this stop sign is affecting the vehicle
                return stop_id, get_bbox_3d_as_dict(stop.trigger_volume, stop.get_transform(), ego_transform)["loc"].x

    return None, None
