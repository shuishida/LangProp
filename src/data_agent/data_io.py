import json
import os.path
from pathlib import Path

import carla
import numpy as np
from PIL import Image
import json


def to_json(obj, indent=4, level=0):
    """
    Faster than jsbeautifier
    """
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        elements = [f"{' ' * (level + 1) * indent}{json.dumps(str(k))}: {to_json(v, indent, level + 1)}" for k, v in obj.items()]
        return '{\n' + ',\n'.join(elements) + ("\n" if elements else '') + ' ' * level * indent + "}"
    try:
        return json.dumps(obj, separators=(', ', ': '))
    except TypeError:
        return json.dumps(str(obj), separators=(', ', ': '))


def serialize_data(data, remove_keys=("ref", "waypoint")):
    if isinstance(data, dict):
        for key in remove_keys:
            if key in data:
                del data[key]
        return {k: serialize_data(v) for k, v in data.items()}
    elif isinstance(data, carla.Vector3D):
        return round(data.x, 5), round(data.y, 5), round(data.z, 5)
    elif isinstance(data, np.ndarray):
        if data.dtype in (np.float64, np.float32):
            data = data.astype(float)
        return data.tolist()
    return data


def deserialize_data(data, as_carla_objects=False):
    if isinstance(data, dict):
        try:
            data = {int(k): v for k, v in data.items()}
        except ValueError:
            pass
        return {k: deserialize_data(v, as_carla_objects) for k, v in data.items()}
    elif isinstance(data, list):
        if as_carla_objects and len(data) == 3 and isinstance(data[0], float) and isinstance(data[1],
                                                                                             float) and isinstance(
                data[2], float):
            return carla.Vector3D(x=data[0], y=data[1], z=data[2])
        else:
            try:
                _data = np.array(data)
                return data if isinstance(_data.dtype, np.object) else _data
            except Exception as e:
                return [deserialize_data(elem, as_carla_objects) for elem in data]
    return data


def load_data(
        data_dir,
        index,
        load_jpg=("rgb", "rgb_high_res"),
        load_png=("birdview", "seg", "depth", "topdown"),
        load_npy=("lidar_topdown",),
        load_json=("bbox", "agent_info", "actors_info"),
        lazy_load_image=False,
        as_carla_objects=False
):
    data_dir = Path(data_dir)
    data = {}

    for data_type in load_jpg + load_png:
        extension = 'jpg' if data_type in load_jpg else 'png'
        data_path = data_dir / data_type / f"{index:04}.{extension}"

        if os.path.exists(data_path):
            if lazy_load_image:
                data[data_type] = data_path
            else:
                data[data_type] = np.array(Image.open(data_path))

    for data_type in load_npy:
        data_path = data_dir / data_type / f"{index:04}.npy"

        if os.path.exists(data_path):
            data[data_type] = np.load(data_dir / data_type / f"{index:04}.npy")

    for data_type in load_json:
        data_path = data_dir / data_type / f"{index:04}.json"

        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data[data_type] = deserialize_data(json.load(f), as_carla_objects)

    return data
