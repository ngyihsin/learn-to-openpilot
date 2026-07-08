"""Day 32 homework — the ML framing: model, loss, generalization.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

The five functions below form one pipeline: split data into train/validation, fit polynomial
models of different complexity, score them with mean-squared error, and *select* the degree that
generalizes best (lowest validation error) rather than the one that memorizes (lowest train error).
"""
from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    """Mean squared error between two 1-D sequences. Returns a plain float."""
    # TODO: convert both to float arrays, return the mean of (y_true - y_pred) ** 2
    raise NotImplementedError


def train_val_split(X, y, val_frac: float = 0.2, seed: int = 0):
    """Shuffle and split into (X_train, y_train, X_val, y_val).

    - The split must be deterministic given ``seed``.
    - Each row's label in ``y`` must stay paired with its features in ``X``.
    - The validation set gets ``round(len(X) * val_frac)`` rows; train gets the rest.
    """
    # TODO:
    #   1. make an RNG:  rng = np.random.default_rng(seed)
    #   2. permute the indices 0..n-1
    #   3. take the first n_val as validation, the rest as train
    #   4. index BOTH X and y with those indices and return the four arrays
    raise NotImplementedError


def fit_polynomial(x, y, degree: int):
    """Fit a polynomial of the given degree to (x, y). Return its coefficients
    (highest power first, i.e. the same order ``np.polyval`` expects)."""
    # TODO: use np.polyfit
    raise NotImplementedError


def predict_polynomial(coeffs, x):
    """Evaluate the polynomial with the given coefficients at the points ``x``."""
    # TODO: use np.polyval
    raise NotImplementedError


def select_degree(x_tr, y_tr, x_val, y_val, degrees) -> int:
    """Return the degree from ``degrees`` whose model, fit on the TRAIN data,
    has the smallest mean-squared error on the VALIDATION data."""
    # TODO: for each degree: fit on train, predict on val, compute mse; keep the argmin.
    raise NotImplementedError
