import math

from torch.utils.data import Dataset


class FactorialDataset(Dataset):
    def __init__(self, size=500):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = index
        label = math.factorial(data)
        return data, label
