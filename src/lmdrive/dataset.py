import random
from typing import Iterator, OrderedDict

import numpy as np
from torch.utils.data import DataLoader, Sampler

from data_agent.dataset.dataset import CarlaDataset


def parse_bbox(bbox):
    results = {}
    for actor_type, actors_bbox in bbox["bbox"].items():
        results[actor_type] = {}
        for actor_id, actor_bbox in actors_bbox.items():
            results[actor_type][actor_id] = actor_bbox["four"]
    return results


class DrivingLMDataset(CarlaDataset):
    def __init__(
            self,
            root,
            load_jpg=(),
            load_png=(),
            load_npy=(),
            load_json=("bbox", "agent_info", "actors_info"),
            as_carla_objects=False,
            infraction_lookahead: int = 0
    ):
        super().__init__(root, load_jpg, load_png, load_npy, load_json, as_carla_objects)
        for i in range(len(self.data)):
            self.data[i]["infractions_ahead"] = []

        if infraction_lookahead:
            # Infractions happen typically a couple of frames after a wrong decision. Look ahead a few frames and
            # if there is an infraction propagate that information.
            for i in range(len(self.data)):
                if self.data[i]["agent_info"]["infractions"]:
                    for j in range(max(0, i - infraction_lookahead), i):
                        if self.data[i]["data_dir"] == self.data[j]["data_dir"]:
                            self.data[j]["infractions_ahead"].extend(self.data[i]["agent_info"]["infractions"])

        self.infractions_ahead = [bool(item["infractions_ahead"]) for item in self.data]

    def transforms(self):
        return {}


def duplicate_infraction_indices(infractions, shuffle: bool, infraction_weight: int):
    indices = []
    for index, infraction in enumerate(infractions):
        if infraction:
            indices.extend([index] * int(infraction_weight))
        else:
            indices.append(index)
    if shuffle:
        random.shuffle(indices)
    return indices


def prioritised_sampling(replay_buffer, infraction_weight, n_samples: int = None):
    indices = duplicate_infraction_indices([item["infractions_ahead"] for item in replay_buffer], shuffle=True, infraction_weight=infraction_weight)
    # sample without replacements
    indices = list(OrderedDict.fromkeys(indices))[:n_samples]
    return indices


def get_batch_samples(train_dataset, replay_buffer, infraction_weight, train_batch_size, replay_batch_size):
    sampled_train_data = list(np.random.choice(train_dataset, train_batch_size)) if train_dataset else []
    sampled_replay_data = [replay_buffer[i] for i in prioritised_sampling(replay_buffer, infraction_weight, replay_batch_size)]
    print(len(sampled_train_data), len(sampled_replay_data))
    batch = sampled_replay_data + sampled_train_data
    return batch


class DrivingDataSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, dataset: DrivingLMDataset, batch_size: int, shuffle: bool = False, infraction_weight: int = 1) -> None:
        super().__init__(dataset)
        self.batch_size = batch_size
        self.indices = duplicate_infraction_indices(dataset.infractions_ahead, shuffle, infraction_weight)
        self._len = len(self.indices)

    def __iter__(self) -> Iterator[int]:
        n = len(self.indices)
        for i in range(0, n, self.batch_size):
            yield from self.indices[i:i + self.batch_size]

    def __len__(self):
        return self._len


# train_sampler = DrivingDataSampler(train_dataset, train_batch_size, shuffle=True, infraction_weight=infraction_weight)
# train_loader = DataLoader(train_dataset, train_batch_size, sampler=train_sampler, num_workers=0, collate_fn=lambda x: x)


def make_carla_lm_dataset(data_root, batch_size, shuffle=False, num_workers: int = 0):
    dataset = DrivingLMDataset(data_root)
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x)
    return data_loader, dataset
