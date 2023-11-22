import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sudoku import Sudoku
from torch.utils.data import DataLoader

from langprop.examples.sudoku.dataset import SudokuDataset
from langprop.module import LPModule, RunConfig
from langprop.trainer import LPTrainer


class Trainer(LPTrainer):
    def preprocess(self, data):
        puzzle, solution, width, height = data
        return (puzzle, width, height), None, (solution, width, height)

    def score(self, result, labels) -> float:
        solution, width, height = labels
        puzzle = Sudoku(width, height, board=result.tolist())
        empty_count = (result == 0).sum()
        correct = puzzle.validate() and empty_count == 0
        return float(correct)


def test_run(run_name: str = "sudoku", batch_size: int = 10, epochs=10):
    train_dataset = SudokuDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: x)
    model = LPModule.from_template(name="solve_sudoku", root=Path(__file__).parent)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + run_name
    trainer = Trainer(model, RunConfig(run_name=run_name, max_keep=2, forward_timeout=200))
    trainer.fit(train_loader, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="sudoku")
    args = parser.parse_args()

    test_run(args.run_name)
