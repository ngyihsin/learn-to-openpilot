"""Day 50 reference solution — evaluation metrics done right.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained (numpy only).
Binary labels: 1 = positive class, 0 = negative.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    """Return (tp, fp, fn, tn) for binary 0/1 arrays.
    tp: true 1 & pred 1 ; fp: true 0 & pred 1 ; fn: true 1 & pred 0 ; tn: true 0 & pred 0."""
    t = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(np.sum((t == 1) & (p == 1)))
    fp = int(np.sum((t == 0) & (p == 1)))
    fn = int(np.sum((t == 1) & (p == 0)))
    tn = int(np.sum((t == 0) & (p == 0)))
    return tp, fp, fn, tn


def precision(tp: int, fp: int) -> float:
    """Of everything you flagged positive, what fraction were right? tp / (tp + fp). 0 if none flagged."""
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    """Of all the real positives, what fraction did you catch? tp / (tp + fn). 0 if no real positives."""
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(prec: float, rec: float) -> float:
    """Harmonic mean of precision and recall: 2*p*r / (p + r). 0 if both are 0."""
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else 0.0
