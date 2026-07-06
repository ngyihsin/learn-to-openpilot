"""Day 31 reference solution — numpy & array thinking.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.
"""
from __future__ import annotations

import numpy as np


def scale_and_shift(x, a: float, b: float):
    x = np.asarray(x, dtype=float)
    return a * x + b


def normalize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def select_positive(x):
    x = np.asarray(x, dtype=float)
    return x[x > 0]


def row_means(M):
    M = np.asarray(M, dtype=float)
    return M.mean(axis=1)


def closest_index(x, value: float) -> int:
    x = np.asarray(x, dtype=float)
    return int(np.argmin(np.abs(x - value)))


def reproducible_randoms(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.random(n)


def reorder(values, order):
    values = np.asarray(values, dtype=float)
    return values[np.asarray(order)]
