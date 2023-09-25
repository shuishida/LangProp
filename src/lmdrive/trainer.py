import json

import carla

from langprop.trainer import LPTrainer
import numpy as np


def preprocess_data(data, with_label: bool = True):
    processed = {}
    # for image_type in self.load_jpg:
    #     data[image_type] = np.array(Image.open(data[image_type]))

    ego_loc = data["ego_info"]["loc"]
    ego_ori = data["ego_info"]["ori"]
    ego_extent = data["ego_info"]["extent"]
    target_loc = data["target"]["near_pos"]
    processed["ego_location_world_coord"] = np.array([ego_loc.x, ego_loc.y]) if isinstance(ego_loc, carla.Vector3D) else np.array([ego_loc[0], ego_loc[1]])
    processed["ego_target_location_world_coord"] = np.array([target_loc.x, target_loc.y]) if isinstance(target_loc, carla.Vector3D) else np.array([target_loc[0], target_loc[1]])
    processed["ego_orientation_unit_vector"] = np.array([ego_ori.x, ego_ori.y]) if isinstance(ego_ori, carla.Vector3D) else np.array([ego_ori[0], ego_ori[1]])
    processed["ego_forward_speed"] = data["ego_info"]["speed"]
    processed["ego_length"] = ego_extent.x if isinstance(ego_extent, carla.Vector3D) else ego_extent[0]
    processed["ego_width"] = ego_extent.y if isinstance(ego_extent, carla.Vector3D) else ego_extent[1]

    # red_light_distance = data["control"]["expert"]["hazards"]["red_light"].values()
    # processed["distance_to_red_light"] = next(iter(red_light_distance)) if len(red_light_distance) else None
    # stop_sign_distance = data["control"]["expert"]["hazards"]["stop_sign"].values()
    # processed["distance_to_stop_sign"] = next(iter(stop_sign_distance)) if len(stop_sign_distance) else None

    processed["distance_to_red_light"] = data["ego_info"]["red_light_distance"]
    processed["distance_to_stop_sign"] = data["ego_info"]["stop_sign_distance"]

    processed["vehicles"] = {}
    for actor_id, actor_info in data["actors_info"]["vehicle"].items():
        loc = actor_info["abs"]["loc"]
        ori = actor_info["abs"]["ori"]
        extent = actor_info["abs"]["extent"]
        processed["vehicles"][actor_id] = {
            "location_world_coord": np.array([loc.x, loc.y]) if isinstance(loc, carla.Vector3D) else np.array([loc[0], loc[1]]),
            "orientation_unit_vector": np.array([ori.x, ori.y]) if isinstance(ori, carla.Vector3D) else np.array([ori[0], ori[1]]),
            "forward_speed": actor_info["speed"],
            "forward_length": extent.x if isinstance(extent, carla.Vector3D) else extent[0],
            "sideways_width": extent.y if isinstance(extent, carla.Vector3D) else extent[1],
        }

    processed["pedestrians"] = {}
    for actor_id, actor_info in data["actors_info"]["walker"].items():
        loc = actor_info["abs"]["loc"]
        ori = actor_info["abs"]["ori"]
        extent = actor_info["abs"]["extent"]
        processed["pedestrians"][actor_id] = {
            "location_world_coord": np.array([loc.x, loc.y]) if isinstance(loc, carla.Vector3D) else np.array([loc[0], loc[1]]),
            "orientation_unit_vector": np.array([ori.x, ori.y]) if isinstance(ori, carla.Vector3D) else np.array([ori[0], ori[1]]),
            "forward_speed": actor_info["speed"],
            "forward_length": extent.x if isinstance(extent, carla.Vector3D) else extent[0],
            "sideways_width": extent.y if isinstance(extent, carla.Vector3D) else extent[1],
        }

    if with_label:
        return processed, dict(
            speed_level=data["control"]["expert"]["speed_level"],
            angle=data["control"]["expert"]["angle"],
            infractions_ahead=data["infractions_ahead"],
            driver_speed_level=data["control"].get("main", {}).get("speed_level"),
        )
    else:
        return processed


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class LMDriveTrainer(LPTrainer):

    def preprocess(self, data):
        inputs, label = preprocess_data(data)
        return (inputs,), None, label

    def score(self, result, labels) -> float:
        pred_speed_level, pred_turn_angle = result
        if labels["infractions_ahead"]:
            if labels["speed_level"] == labels["driver_speed_level"]:
                return 0
            elif pred_speed_level == labels["speed_level"]:
                return 1
            elif pred_speed_level == labels["driver_speed_level"]:
                return self.run_config.save_config["infraction_penalty"]
            else:
                return 0
        elif pred_speed_level == labels["speed_level"]:
            return 1
        else:
            return 0

    def test_output(self, output, func_args, func_kwargs, label):
        speed_level, pred_turn_angle = output
        gt_turn_angle = label["angle"]
        turn_error_margin = 10
        assert speed_level in ("MOVE", "SLOW", "STOP"), f"Invalid speed_level: {speed_level}"
        assert isinstance(pred_turn_angle, (int, float)), "Steering angle should be in float"
        assert -180 < pred_turn_angle < 180, "Steering angle should be in the range of -180 to 180 degrees"
        assert abs(pred_turn_angle - gt_turn_angle) < turn_error_margin, \
            (f"Predicted turn angle {pred_turn_angle} has an error with the ground truth "
             f"turn angle {gt_turn_angle} by more than {turn_error_margin} degrees.")
