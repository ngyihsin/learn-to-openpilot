"""Day 34 reference solution — classification: softmax & cross-entropy.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.
"""
from __future__ import annotations

import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def one_hot(label: int, num_classes: int):
    v = np.zeros(num_classes, dtype=float)
    v[label] = 1.0
    return v


def cross_entropy(probs, label: int) -> float:
    probs = np.asarray(probs, dtype=float)
    p = np.clip(probs[label], 1e-12, 1.0)
    return float(-np.log(p))


def predict(probs) -> int:
    return int(np.argmax(np.asarray(probs, dtype=float)))


def accuracy(pred_labels, true_labels) -> float:
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)
    return float(np.mean(pred == true))
