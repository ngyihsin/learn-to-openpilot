"""Day 33 reference solution — regression by gradient descent.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.
"""
from __future__ import annotations

import numpy as np


def predict(w: float, b: float, x):
    x = np.asarray(x, dtype=float)
    return w * x + b


def mse_loss(w: float, b: float, x, y) -> float:
    y = np.asarray(y, dtype=float)
    err = predict(w, b, x) - y
    return float(np.mean(err**2))


def gradients(w: float, b: float, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = predict(w, b, x) - y
    dw = float(np.mean(2 * err * x))
    db = float(np.mean(2 * err))
    return dw, db


def gradient_descent_step(w: float, b: float, x, y, lr: float):
    dw, db = gradients(w, b, x, y)
    return w - lr * dw, b - lr * db


def fit_linear(x, y, lr: float = 0.05, epochs: int = 2000):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        w, b = gradient_descent_step(w, b, x, y, lr)
    return w, b
