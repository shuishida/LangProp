import os
import json
from collections import defaultdict

import tqdm
import numpy as np
import argparse


SUPERVISION_KEYS = ["features", "value", "speed", "action", "action_mu", "action_sigma", "only_ap_brake"]
MEASUREMENT_KEYS = ['x', 'y', 'theta', 'x_target', 'y_target', 'target_command']


def get_seq_input(seq, i, n_input_frames: int):
    return np.array(seq[i - (n_input_frames - 1):i + 1])


def get_seq_future(seq, i, n_future_frames: int):
    return np.array(seq[i + 1:i + n_future_frames + 1])


def listdir_fullpath(d):
    dirs = [os.path.join(d, f) for f in os.listdir(d)]
    return sorted([d for d in dirs if os.path.isdir(d)])


def gen_single_route(route_folder, n_input_frames=1, n_future_frames=4, length=None):
    if not length:
        length = len(os.listdir(os.path.join(route_folder, 'measurements')))
    if length < n_input_frames + n_future_frames:
        return

    data_dict = defaultdict(list)
    supervision_dict = defaultdict(list)
    measurements_dict = defaultdict(list)

    for i in range(length):
        with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
            measurement = json.load(read_file)

        for key in MEASUREMENT_KEYS:
            measurements_dict[key].append(measurement[key])

        roach_supervision_data = np.load(os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"),
                                         allow_pickle=True).item()
        for key in SUPERVISION_KEYS:
            supervision_dict[key].append(roach_supervision_data[key])

    for i in range(n_input_frames - 1, length - n_future_frames):

        data_dict['input_x'].append(get_seq_input(measurements_dict['x'], i, n_input_frames))
        data_dict['input_y'].append(get_seq_input(measurements_dict['y'], i, n_input_frames))
        data_dict['input_theta'].append(get_seq_input(measurements_dict['theta'], i, n_input_frames))

        data_dict['future_x'].append(get_seq_future(measurements_dict['x'], i, n_future_frames))
        data_dict['future_y'].append(get_seq_future(measurements_dict['y'], i, n_future_frames))
        data_dict['future_theta'].append(get_seq_future(measurements_dict['theta'], i, n_future_frames))

        data_dict['x_target'].append(measurements_dict['x_target'][i])
        data_dict['y_target'].append(measurements_dict['y_target'][i])
        data_dict['target_command'].append(measurements_dict["target_command"][i])

        _, route_folder_name = os.path.split(route_folder)
        front_img_list = [os.path.join(route_folder_name, "rgb", f"{str(i - offset).zfill(4)}.png")
                          for offset in reversed(range(n_input_frames))]
        data_dict['input_front_img'].append(front_img_list)

        for key, seq in supervision_dict.items():
            data_dict[key].append(seq[i])
            data_dict[f"future_{key}"].append(get_seq_future(seq, i, n_future_frames))

    return data_dict


def get_valid_traj_length(subfolder_path, record):
    total_length = len(os.listdir(os.path.join(subfolder_path, "measurements")))
    if record["scores"]["score_composed"] >= 100:
        return total_length
    # timeout or blocked, remove the last ones where the vehicle stops
    if record["infractions"]["route_timeout"] or record["infractions"]["vehicle_blocked"]:
        stop_index = 0
        for i in reversed(range(total_length)):
            with open(os.path.join(subfolder_path, "measurements", str(i).zfill(4)) + ".json", 'r') as mf:
                speed = json.load(mf)["speed"]
                if speed > 0.1:
                    stop_index = i
                    break
        stop_index = min(total_length, stop_index + 20)
        return stop_index
    # # collision or red-light
    # elif record["infractions"]["red_light"] or \
    #         record["infractions"]["collisions_pedestrian"] or \
    #         record["infractions"]["collisions_vehicle"] or \
    #         record["infractions"]["collisions_layout"]:
    #     stop_index = max(0, total_length - 10)
    return total_length


def gen_folder(folder_path, n_input_frames=1, n_future_frames=4):
    data_dict = defaultdict(list)

    # read the record of each route
    with open(folder_path.rstrip('/') + ".json", 'r') as f:
        records = json.load(f)
    records = records["_checkpoint"]["records"]

    for subfolder_path, record in tqdm.tqdm(zip(listdir_fullpath(folder_path), records)):
        length = get_valid_traj_length(subfolder_path, record)
        seq_data = gen_single_route(subfolder_path, n_input_frames, n_future_frames, length)
        if seq_data:
            for key, value in seq_data.items():
                data_dict[key].extend(value)

    np_data_dict = {}
    for k, v in data_dict.items():
        np_data_dict[k] = np.array(v)

    file_path = os.path.join(folder_path, "packed_data")
    np.save(file_path, np_data_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to data directory")

    args = parser.parse_args()

    gen_folder(args.data)
