from dataclasses import dataclass, field
from typing import Union, Optional

import numpy as np

AUTO = "auto"


@dataclass
class SensorConfig:
    x: float
    y: float
    z: float


@dataclass
class CameraConfig(SensorConfig):
    width: int
    height: int
    fov: Optional[Union[float, int]] = None
    focal_length: Optional[float] = None
    intrinsics: np.ndarray = field(init=False)

    def __post_init__(self):
        assert not (self.fov and self.focal_length), "fov and focal length cannot be declared at the same time"
        if self.fov:
            self.focal_length = self.width / (2.0 * np.tan(np.radians(self.fov / 2)))
        if self.focal_length:
            self.fov = np.degrees(np.arctan(self.width / self.focal_length / 2.0)) * 2
        self.intrinsics = self.get_intrinsics()

    def get_intrinsics(self):
        """
        returns the calibration matrix K for the given sensor
        """
        return np.array([
            [self.focal_length, 0, self.width / 2.0],
            [0, self.focal_length, self.height / 2.0],
            [0, 0, 1]
        ])


def get_sensors(camera_config: CameraConfig, high_res_config: CameraConfig = None,
                topdown_config: CameraConfig = None, lidar_config: SensorConfig = None,
                sensor_tick: float = 0.0, use_front_only: bool = True, side_angle=90,
                use_rgb: bool = True, use_seg: bool = False, use_depth: bool = False, use_map: bool = False):
    assert not (camera_config is None and (use_seg or use_depth)), \
        "Camera config must be provided if you are requesting semantic segmentation or depth"

    sensors = [
        {
            'type': 'sensor.other.imu',
            'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'id': 'imu'
        },
        {
            'type': 'sensor.other.gnss',
            'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'id': 'gps'
        },
        {
            'type': 'sensor.speedometer',
            'id': 'speed'
        }
    ]
    if use_rgb:
        sensors += [
            {
                'type': 'sensor.camera.rgb',
                'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                'sensor_tick': sensor_tick,
                'id': "rgb_front"
            },
        ]
        if not use_front_only:
            sensors += [
                {
                    'type': 'sensor.camera.rgb',
                    'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -side_angle,
                    'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                    'sensor_tick': sensor_tick,
                    'id': 'rgb_left'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': side_angle,
                    'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                    'sensor_tick': sensor_tick,
                    'id': 'rgb_right'
                },
            ]
    if high_res_config:
        sensors += [
            {
                'type': 'sensor.camera.rgb',
                'x': high_res_config.x, 'y': high_res_config.y, 'z': high_res_config.z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': high_res_config.width, 'height': high_res_config.height, 'fov': high_res_config.fov,
                'sensor_tick': sensor_tick,
                'id': 'rgb_high_res'
            },
        ]
    if use_seg:
        sensors += [
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                'sensor_tick': sensor_tick,
                'id': 'seg_front'
            },
        ]
        if not use_front_only:
            sensors += [
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -side_angle,
                    'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                    'sensor_tick': sensor_tick,
                    'id': 'seg_left'
                },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': side_angle,
                    'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                    'sensor_tick': sensor_tick,
                    'id': 'seg_right'
                },
            ]
    if use_depth:
        sensors += [
            {
                'type': 'sensor.camera.depth',
                'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                'sensor_tick': sensor_tick,
                'id': 'depth_front'
            },
        ]
        if not use_front_only:
            sensors += [
                {
                    'type': 'sensor.camera.depth',
                    'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -side_angle,
                    'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                    'sensor_tick': sensor_tick,
                    'id': 'depth_left'
                },
                {
                    'type': 'sensor.camera.depth',
                    'x': camera_config.x, 'y': camera_config.y, 'z': camera_config.z,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': side_angle,
                    'width': camera_config.width, 'height': camera_config.height, 'fov': camera_config.fov,
                    'sensor_tick': sensor_tick,
                    'id': 'depth_right'
                },
            ]
    if use_map:
        sensors += [
            {
                'type': 'sensor.opendrive_map',
                'reading_frequency': 1e-6,
                'id': 'hd_map'
            },
        ]
    if lidar_config:
        # In the default leaderboard setting, the lidar rotation frequency is set to 10, meaning that with 20 FPS
        # the lidar will only get captures of 180 degrees at every timestep.
        # Here we collect measurements -90 - 90.
        sensors += [
            {
                "type": "sensor.lidar.ray_cast",
                "x": lidar_config.x, "y": lidar_config.y, "z": lidar_config.z, "roll": 0.0, "pitch": 0.0, "yaw": -90.0,
                # 'sensor_tick': sensor_tick,
                "id": "lidar",
            },
        ]
    if topdown_config:
        sensors += [
            {
                "type": "sensor.camera.semantic_segmentation",
                "x": topdown_config.x, "y": topdown_config.y, "z": topdown_config.z,
                "roll": 0.0, "pitch": -90.0, "yaw": 0.0,
                "width": topdown_config.width, "height": topdown_config.height, "fov": topdown_config.fov,
                'sensor_tick': sensor_tick,
                "id": "topdown",
            },
        ]
    return sensors
