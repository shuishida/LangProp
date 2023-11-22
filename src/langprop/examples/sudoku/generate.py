from pathlib import Path

from sudoku import Sudoku

import numpy as np


def generate_sudoku_dataset(path, difficulty=0.5, min_size=3, max_size=5, n_samples=100):
    widths = np.random.randint(min_size, max_size, size=n_samples)
    heights = np.random.randint(min_size, max_size, size=n_samples)
    boards = []
    solutions = []
    for i in range(n_samples):
        puzzle = Sudoku(widths[i], heights[i], seed=i).difficulty(difficulty)
        board = np.array(puzzle.board)
        board[board == None] = 0
        board = board.astype(np.int)
        boards.append(board)
        solutions.append(np.array(puzzle.solve().board).astype(np.int))
    data = {"boards": boards, "widths": widths, "heights": heights, "solutions": solutions,
            "n_samples": n_samples, "difficulty": difficulty, "min_size": min_size, "max_size": max_size}
    np.savez(path, **data)
    return data
