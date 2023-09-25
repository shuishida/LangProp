import os
from typing import List

from PIL import Image
import numpy as np
import torch as th
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from agents.navigation.local_planner import RoadOption
from data_agent.data_io import load_data


def apply_transforms(data: dict, transforms: dict):
    """
    Mutable operation
    """
    for key, transform in transforms.items():
        if transform is None:
            if key in data:
                del data[key]
        elif isinstance(transform, dict):
            apply_transforms(data[key], transform)
        else:
            data[key] = transform(data[key])


class CarlaDataset(Dataset):
    def __init__(
            self,
            root,
            load_jpg=("rgb", "rgb_high_res"),
            load_png=("birdview", "seg", "depth", "topdown"),
            load_npy=("lidar_topdown",),
            load_json=("agent_info",),
            as_carla_objects=False,
    ):
        print("Loading dataset located at: ", root)
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset not found: {root}")
        self.root = root
        self.load_jpg = load_jpg
        self.load_png = load_png
        self.load_npy = load_npy
        self.load_json = load_json
        self.as_carla_objects = as_carla_objects

        find_dir = (load_json + load_npy + load_jpg + load_png)[0]

        self.data: List[dict] = []
        for path, dirs, files in os.walk(root, topdown=False):
            if find_dir in dirs:
                for index in range(len(os.listdir(os.path.join(path, find_dir)))):
                    self.data.append(self.load_data(path, index))

        self._len = len(self.data)

    def transforms(self):
        return {
            "agent_info": {
                "weather": None,
                "ego_info": None,
                "hazards": None,
                "target": {
                    "near_command": lambda x: COMMAND_TO_ONEHOT[x],
                    "query_command": lambda x: COMMAND_TO_ONEHOT[x],
                }
            },
        }

    def load_data(self, data_dir, index):
        data = load_data(data_dir, index, load_jpg=self.load_jpg, load_png=self.load_png, load_npy=self.load_npy,
                         load_json=self.load_json, lazy_load_image=True)
        apply_transforms(data, self.transforms())
        return {**data, "index": index, "data_dir": data_dir}

    def __len__(self):
        """Returns the length of the dataset. """
        return self._len

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = self.data[index]

        data = {**data, **data["agent_info"]}
        del data["agent_info"]

        for image_type in self.load_jpg + self.load_png:
            data[image_type] = np.array(Image.open(data[image_type]))

        return data


def collate_list_fn(batch, *, collate_fn_map = None):
    batch = [collate_fn(elem) for elem in batch]
    elem = batch[0]
    if isinstance(elem, th.Tensor):
        try:
            return th.stack(batch, 0)
        except:
            print(batch)
            raise Exception
    return batch


def collate_float_fn(batch, *, collate_fn_map = None):
    return th.tensor(batch, dtype=th.float)


def collate_fn(batch):
    return collate(batch, collate_fn_map={**default_collate_fn_map,
                                          list: collate_list_fn,
                                          float: collate_float_fn})


COMMAND_TO_ONEHOT = {str(e): th.eye(len(RoadOption))[i] for i, e in enumerate(RoadOption)}
