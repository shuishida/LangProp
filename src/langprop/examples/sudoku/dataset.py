import os.path
from pathlib import Path

from sudoku import Sudoku

from torch.utils.data import Dataset
import numpy as np

from langprop.examples.sudoku.generate import generate_sudoku_dataset


class SudokuDataset(Dataset):
    def __init__(self, path=Path(__file__).parent, difficulty=0.5, min_size=3, max_size=5, n_samples=100):
        filepath = Path(path) / f"sudoku_{difficulty}_{min_size}_{max_size}_{n_samples}.npz"
        if os.path.exists(filepath):
            self.data = np.load(filepath, allow_pickle=True)
        else:
            self.data = generate_sudoku_dataset(filepath, difficulty=difficulty,
                                                min_size=min_size, max_size=max_size, n_samples=n_samples)

    def __len__(self):
        return self.data["n_samples"]

    def __getitem__(self, index):
        return self.data["boards"][index], self.data["solutions"][index], self.data["widths"][index], self.data["heights"][index]
