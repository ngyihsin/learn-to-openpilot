"""Day 33 homework — regression by gradient descent.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

You are fitting a straight line ``y = w * x + b`` to data — but instead of a formula, you find
``w`` and ``b`` by gradient descent: predict, measure the loss, compute which way is downhill, and
take a small step. That loop is the engine behind every neural network.
"""
from __future__ import annotations

import numpy as np


def predict(w: float, b: float, x):
    """Return the model's predictions ``w * x + b`` for every point in ``x`` (a 1-D array)."""
    # TODO
    raise NotImplementedError


def mse_loss(w: float, b: float, x, y) -> float:
    """Mean squared error of the line (w, b) on the data (x, y). Return a plain float."""
    # TODO: predict, then average the squared errors
    raise NotImplementedError


def gradients(w: float, b: float, x, y):
    """Return (dL/dw, dL/db), the partial derivatives of the MSE loss at (w, b).

        dL/dw = mean(2 * (pred - y) * x)
        dL/db = mean(2 * (pred - y))
    """
    # TODO: compute pred, the error (pred - y), then the two gradients. Return them as a tuple.
    raise NotImplementedError


def gradient_descent_step(w: float, b: float, x, y, lr: float):
    """Take ONE gradient-descent step and return the updated (w, b).
    Move *against* the gradient: new = old - lr * grad."""
    # TODO: get the gradients, then step both parameters downhill
    raise NotImplementedError


def fit_linear(x, y, lr: float = 0.05, epochs: int = 2000):
    """Fit a line to (x, y) by gradient descent, starting from w = 0.0, b = 0.0.
    Run ``epochs`` steps and return the final (w, b)."""
    # TODO: initialise w, b to 0.0; loop epochs times calling gradient_descent_step; return (w, b)
    raise NotImplementedError
