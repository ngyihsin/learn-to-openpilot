"""Day 32 reference solution — the ML framing.

Run the grader against this with ``LP_IMPL=solution pytest -q`` to confirm the tests are correct.
Self-contained: no imports from ``homework``.
"""
from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def train_val_split(X, y, val_frac: float = 0.2, seed: int = 0):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def fit_polynomial(x, y, degree: int):
    return np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), degree)


def predict_polynomial(coeffs, x):
    return np.polyval(coeffs, np.asarray(x, dtype=float))


def select_degree(x_tr, y_tr, x_val, y_val, degrees) -> int:
    best_degree, best_err = None, None
    for d in degrees:
        coeffs = fit_polynomial(x_tr, y_tr, d)
        err = mse(y_val, predict_polynomial(coeffs, x_val))
        if best_err is None or err < best_err:
            best_degree, best_err = d, err
    return best_degree
