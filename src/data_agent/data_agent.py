import math
import os
from typing import Dict

import cv2
import carla
import py_trees
import yaml
from carla_birdeye_view import BirdViewProducer, PixelDimensions, BirdViewCropType

from data_agent.core_agent import CoreAgent
from data_agent.process_data import collect_actors_info, collect_agent_info, get_nearby_objects, convert_depth, BBox, get_image_bboxes, get_bbox_mask, \
    lidar_to_histogram_features, get_traffic_light_waypoints, get_traffic_light_triggers, \
    follow_waypoint_to_intersect
from data_agent.data_io import serialize_data, to_json
from data_agent.render import draw_points_on_image, CITYSCAPES_ID
from data_agent.visualize import gen_debug_images, birdview_compress
from carla_tools.map_utils import gps_to_pos
from carla_tools.sensors import CameraConfig, get_sensors, SensorConfig
from carla_tools.planner import RoutePlanner, to_relative_pos

import numpy as np
from PIL import Image

from leaderboard.scenarios.scenarioatomics.atomic_criteria import ActorSpeedAboveThresholdTest
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, InRouteTest, \
    OutsideRouteLanesTest, RunningRedLightTest, RunningStopTest


def get_entry_point():
    return "DataAgent"


class DataAgent(CoreAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.camera_config = CameraConfig(**self.config["sensor"]["camera"])
        self.high_res_config = CameraConfig(**self.config["sensor"]["high_res"]) if "high_res" in self.config["sensor"] else None
        self.topdown_config = CameraConfig(**self.config["sensor"]["topdown"])
        self.lidar_config = SensorConfig(**self.config["sensor"]["lidar"]) if "lidar" in self.config["sensor"] else None
        self.front_only = self.config["sensor"]["front_only"]
        self.use_seg = self.config["sensor"].get("use_seg", True)
        self.side_angle = self.config["sensor"].get("side_angle")

        self.camera_poses = ['front'] if self.front_only else ['left', 'front', 'right']

        self.save_agent_info = ['measurements', 'target', 'ego_info', 'infractions', "control"]
        self.save_json = ["bbox", "agent_info", "actors_info"]
        self.save_jpg = ["rgb", "debug"]
        if self.high_res_config:
            self.save_jpg += ["rgb_high_res"]
        self.save_png = ["birdview", "topdown", "depth"]
        if self.use_seg:
            self.save_png += ["seg"]
        self.save_npy = ["lidar", "lidar_topdown"] if self.lidar_config else []

    def _init(self):
        """
        Called at the start of run_step()
        """
        if self.save_path:
            for data_type in (self.save_npy + self.save_jpg + self.save_json + self.save_png):
                (self.save_path / data_type).mkdir(parents=True, exist_ok=True)

        self._command_planner = RoutePlanner(self._global_plan, 7.5)
        self._waypoint_planner = RoutePlanner(self._global_plan_dense, 4.0, fov=180)

        print(f"Number of waypoints in route: {len(self._waypoint_planner.route)}")

        self._sensors = self.sensor_interface._sensors_objects
        self._world = CarlaDataProvider.get_world()
        self._map = self._world.get_map()
        self._vehicle = CarlaDataProvider.get_hero_actor()

        pixels_per_meter = self.topdown_config.focal_length / self.topdown_config.z
        assert abs(pixels_per_meter - int(pixels_per_meter)) < 0.01, \
            f"Please change the topdown config such that pixels per meter ({pixels_per_meter}) is an integer value."

        self.birdview_producer = BirdViewProducer(
            CarlaDataProvider.get_client(),  # carla.Client
            target_size=PixelDimensions(width=self.topdown_config.width, height=self.topdown_config.height),
            pixels_per_meter=int(pixels_per_meter),
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )

        self._yaw_when_brake = None

        self.criteria = self.get_criteria()
        self.infraction_to_step = {}

    def get_criteria(self):
        route = CarlaDataProvider.get_ego_vehicle_route()

        collision_criterion = CollisionTest(self._vehicle, terminate_on_failure=False)
        route_criterion = InRouteTest(self._vehicle,
                                      route=route,
                                      offroad_max=25,
                                      terminate_on_failure=False)
        outsidelane_criterion = OutsideRouteLanesTest(self._vehicle, route=route)
        red_light_criterion = RunningRedLightTest(self._vehicle)
        stop_criterion = RunningStopTest(self._vehicle)
        blocked_criterion = ActorSpeedAboveThresholdTest(self._vehicle,
                                                         speed_threshold=0.1,
                                                         below_threshold_max_time=85.0,
                                                         terminate_on_failure=False)
        return dict(
            collision=collision_criterion,
            route=route_criterion,
            outsidelane=outsidelane_criterion,
            red_light=red_light_criterion,
            stop=stop_criterion,
            # blocked=blocked_criterion,
        )

    def sensors(self):
        return get_sensors(self.camera_config, self.high_res_config, self.topdown_config, self.lidar_config,
                           sensor_tick=self.eval_freq / self.config["carla_fps"],
                           use_front_only=self.front_only, side_angle=self.side_angle, use_seg=self.use_seg, use_depth=True)

    def process(self, input_data, timestamp):
        self._actors = self._world.get_actors()
        vehicles = get_nearby_objects(self._vehicle, self._actors.filter("*vehicle*"),
                                      cutoff_distance=self.config["distance"]["vehicle"])
        walkers = get_nearby_objects(self._vehicle, self._actors.filter("*walker*"),
                                     cutoff_distance=self.config["distance"]["walker"])
        traffic_lights = get_nearby_objects(self._vehicle, self._actors.filter("*traffic_light*"),
                                            cutoff_distance=self.config["distance"]["traffic_light"])
        stop_signs = get_nearby_objects(self._vehicle, self._actors.filter("*stop*"),
                                        cutoff_distance=self.config["distance"]["stop_sign"])

        self.nearby_actors = dict(
            vehicle={k: v for k, v in vehicles.items() if (k != self._vehicle.id and v.is_alive)},
            walker=walkers,
            traffic_light=traffic_lights,
            stop_sign=stop_signs
        )

        self.light_waypoints = get_traffic_light_waypoints(traffic_lights, self._map)

        gps = input_data['gps'][:2]
        ego_pos = gps_to_pos(*gps)
        ego_transform = self._vehicle.get_transform()
        ego_loc_np = np.array([ego_transform.location.x, ego_transform.location.y])
        assert np.linalg.norm(ego_loc_np - ego_pos) < 0.5, \
            f"Ego location {ego_loc_np} and position calculated from GPS {ego_pos}" \
            f"do not match, possibly due to a different lat long reference."

        speed = input_data['speed']['speed']
        compass = input_data['imu'][-1]
        if math.isnan(compass):  # simulation bug
            compass = 0.0
        # compass is measured against north, but in Carla x is east and y is south
        compass -= np.pi / 2  # 90 degrees

        measurements = dict(
            gps=gps,
            pos=ego_pos,
            speed=speed,
            theta=compass,
            timestamp=timestamp,
        )

        near_pos, near_vec, near_command = self._waypoint_planner.run_step(ego_pos, compass)
        query_pos, query_vec, query_command = self._command_planner.run_step(ego_pos, compass)

        n_waypoints = self.config["n_waypoints"]
        waypoints = [pos for pos, _ in self._waypoint_planner.route[:n_waypoints]]
        if len(waypoints) < n_waypoints:
            waypoints += [waypoints[-1]] * (n_waypoints - len(waypoints))
        waypoints = to_relative_pos(np.array(waypoints), ego_pos, compass)

        target = dict(
            near_pos=near_pos,
            near_vec=near_vec,
            near_command=near_command,
            query_pos=query_pos,
            query_vec=query_vec,
            query_command=query_command,
            waypoints=waypoints
        )

        ego_info = collect_agent_info(self._vehicle, self.light_waypoints, stop_signs, self._map)
        actors_info = collect_actors_info(self._vehicle, self.nearby_actors, self.light_waypoints, self._map)

        birdview = birdview_compress(self.birdview_producer.produce(agent_vehicle=self._vehicle))

        bboxes_all = {}
        rgb_list = []
        depth_list = []

        for cam_pos in self.camera_poses:
            rgb_img = cv2.cvtColor(input_data['rgb_' + cam_pos][:, :, :3], cv2.COLOR_BGR2RGB)
            depth_img = input_data['depth_' + cam_pos][:, :, :3]

            bboxes = self.get_bboxes(depth_img, 'depth_' + cam_pos)

            rgb_list.append(rgb_img)
            depth_list.append(depth_img)
            bboxes_all[cam_pos] = bboxes

        rgb_img = np.concatenate(rgb_list, axis=1)
        depth_img = np.concatenate(depth_list, axis=1)

        infractions = []
        for criterion_name, criterion in self.criteria.items():
            criterion.update()
            for e in criterion.list_traffic_events:
                event_info = (e.get_type().name, e.get_message())
                if event_info in self.infraction_to_step:
                    if (self.infraction_to_step[event_info] - 1) // self.save_freq == (self.step - 1) // self.save_freq:
                        # include all infractions that happened after the last save occurred.
                        infractions.append(event_info)
                else:
                    infractions.append(event_info)
                    self.infraction_to_step[event_info] = self.step

        results = dict(
            measurements=measurements,
            target=target,
            ego_info=ego_info,
            actors_info=actors_info,
            birdview=birdview,
            rgb=rgb_img,
            depth=depth_img,
            bbox=bboxes_all,
            infractions=infractions
        )
        if "lidar" in input_data:
            results["lidar"] = lidar = input_data["lidar"]
            results["lidar_topdown"] = lidar_to_histogram_features(lidar, self.lidar_config)
        if self.high_res_config:
            results["rgb_high_res"] = cv2.cvtColor(input_data['rgb_high_res'][:, :, :3], cv2.COLOR_BGR2RGB)

        if "topdown" in input_data:
            topdown = input_data["topdown"][:, :, 2]  # Semantic segmentation is stored in the R channel in BGR
            topdown = self._draw_actors_on_topdown(topdown, near_pos, query_pos)
            results["topdown"] = topdown

        if self.use_seg:
            # Semantics encoded in the R channel in BGR
            seg_list = [np.copy(input_data['seg_' + cam_pos][:, :, 2]) for cam_pos in self.camera_poses]
            seg_img = np.concatenate(seg_list, axis=1)
            results["seg"] = seg_img

        return results

    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)

        if self.config.get("yaw_noise", 0):
            self.add_yaw_noise(control)

        return control

    def add_yaw_noise(self, control):
        ego_transform = self._vehicle.get_transform()
        rotation = ego_transform.rotation

        yaw_range = self.config["yaw_noise"]
        yaw = rotation.yaw + (np.random.uniform() * 2 - 1) * yaw_range

        if control.throttle == 0:
            if not self._yaw_when_brake:
                self._yaw_when_brake = rotation.yaw
            yaw = np.clip(yaw, self._yaw_when_brake - yaw_range, self._yaw_when_brake + yaw_range)
        else:
            self._yaw_when_brake = None

        self._vehicle.set_transform(carla.Transform(location=ego_transform.location,
                                                    rotation=carla.Rotation(
                                                        pitch=rotation.pitch,
                                                        yaw=yaw,
                                                        roll=rotation.roll
                                                    )))

    def save(self, tick_data):
        frame = self.step // self.save_freq

        tick_data["agent_info"] = {key: tick_data[key] for key in self.save_agent_info}
        tick_data["debug"] = gen_debug_images(tick_data)["main"]

        for name in self.save_json:
            filepath = self.save_path / name / f"{frame:04d}.json"
            with open(filepath, "w") as f:
                f.write(to_json(serialize_data(tick_data[name])))

        for name in self.save_jpg:
            filepath = self.save_path / name / f"{frame:04d}.jpg"
            Image.fromarray(tick_data[name]).save(filepath)

        for name in self.save_png:
            filepath = self.save_path / name / f"{frame:04d}.png"
            Image.fromarray(tick_data[name]).save(filepath)

        for name in self.save_npy:
            filepath = self.save_path / name / f"{frame:04d}.npy"
            np.save(filepath, tick_data[name], allow_pickle=True)

        filepath = self.save_path / "infractions.json"
        with open(filepath, "w") as f:
            f.write(to_json(self.infraction_to_step))

        filepath = self.save_path / "config.yaml"
        with open(filepath, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _draw_actors_on_topdown(self, topdown, near_node, query_node):
        topdown_transform = self._sensors['topdown'].get_transform()

        waypoint_locs = np.array([wp.pos for wp in self._waypoint_planner.route])
        topdown = draw_points_on_image(topdown, waypoint_locs, CITYSCAPES_ID["waypoints"], topdown_transform, self.topdown_config, radius=1)
        topdown = draw_points_on_image(topdown, np.array([near_node]), CITYSCAPES_ID["near_pos"], topdown_transform, self.topdown_config)
        topdown = draw_points_on_image(topdown, np.array([query_node]), CITYSCAPES_ID["query_pos"], topdown_transform, self.topdown_config)

        traffic_lights = self.nearby_actors["traffic_light"].values()

        traffic_light_locs = []
        traffic_light_states = []

        for light in traffic_lights:
            trigger_points, trigger_center = get_traffic_light_triggers(light)
            trigger_points.append(light.get_location())
            traffic_light_locs.extend(trigger_points)
            traffic_light_states.extend([light.state.real + 23] * len(trigger_points))

        for wp, light in self.light_waypoints.items():
            traffic_light_locs.append(wp.transform.location)
            traffic_light_states.append(light.state.real + 23)

        topdown = draw_points_on_image(topdown, traffic_light_locs, traffic_light_states,
                                       topdown_transform, self.topdown_config)

        stop_sign_locs = [stop.get_location() for stop in self.nearby_actors["stop_sign"].values()]

        topdown = draw_points_on_image(topdown, stop_sign_locs, CITYSCAPES_ID["stop_sign"], topdown_transform, self.topdown_config)

        loc = self._vehicle.get_location()
        loc_wp = self._map.get_waypoint(loc)
        loc_wp = follow_waypoint_to_intersect(loc_wp)

        topdown = draw_points_on_image(topdown, [loc_wp.transform.location], CITYSCAPES_ID["ego_waypoint"], topdown_transform, self.topdown_config)

        return topdown

    def get_bboxes(self, depth_img, cam_id) -> Dict[str, Dict[str, Dict[int, BBox]]]:
        """
        Returns a dict of 3d and 2d bounding boxes given a list of actors and the camera transform
        Bounding box coordinates will be calculated using the camera_config (not the rgb_camera_config)
        so if you have a different rgb camera setup you would have to scale the results accordingly.
        """
        depth = convert_depth(depth_img)
        camera_transform = self._sensors[cam_id].get_transform()

        bboxes: Dict[str, Dict[int, BBox]] = {}
        for actor_type, actors in self.nearby_actors.items():
            bboxes[actor_type] = get_image_bboxes(actors.values(), depth, camera_transform, self.camera_config)

        triggers: Dict[str, Dict[int, BBox]] = {}
        for actor_type in ["traffic_light", "stop_sign"]:
            actors = self.nearby_actors[actor_type]
            triggers[actor_type] = get_image_bboxes(actors.values(), depth, camera_transform, self.camera_config,
                                                    get_trigger=True)

        return dict(bbox=bboxes, trigger=triggers)

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        for criterion in self.criteria.values():
            criterion.terminate(py_trees.common.Status.INVALID)
        super().destroy()
