from collections import defaultdict
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root")

args = parser.parse_args()

root_dir_all = args.root

train_towns = ['town01', 'town03', 'town04', 'town06', ]
val_towns = ['town02', 'town05', 'town07', 'town10']
train_data, val_data = [], []
for town in train_towns:
    train_data.append(os.path.join(root_dir_all, town))
    train_data.append(os.path.join(root_dir_all, town + '_addition'))
for town in val_towns:
    val_data.append(os.path.join(root_dir_all, town + '_val'))

for data_folders, filename in [(train_data, "train.npy"), (val_data, "val.npy")]:
    concat_data = defaultdict(list)
    for sub_root in data_folders:
        data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()
        for k, v in data.items():
            concat_data[k].extend(v)
    concat_data["x_target"], concat_data["y_target"] = concat_data["y_target"], concat_data["x_target"]
    concat_data["future_x"], concat_data["future_y"] = concat_data["future_y"], concat_data["future_x"]
    concat_data["input_x"], concat_data["input_y"] = concat_data["input_y"], concat_data["input_x"]
    concat_data["features"] = concat_data.pop("feature")
    concat_data["future_features"] = concat_data.pop("future_feature")

    for paths in concat_data.pop("front_img"):
        full_paths = [root_dir_all + path for path in paths]
        concat_data["input_front_img"].append(full_paths)

    np_data_dict = {}
    for k, v in concat_data.items():
        np_data_dict[k] = np.array(v)

    np.save(os.path.join(root_dir_all, filename), np_data_dict)
