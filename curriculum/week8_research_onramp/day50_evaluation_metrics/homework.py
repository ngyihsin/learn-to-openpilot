"""Day 50 homework — evaluation metrics done right.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

"91% accurate" can be worthless — a model that always predicts "no pedestrian" is ~99% accurate on a
street with few pedestrians, and lethal. Real evaluation uses **precision** and **recall**, which you
compute from four counts. Build them here; you'll use them to judge every experiment in Day 51.

Binary labels: 1 = positive class, 0 = negative.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    """Return (tp, fp, fn, tn) for binary 0/1 arrays.
        tp = true 1 & pred 1   (caught a real positive)
        fp = true 0 & pred 1   (false alarm)
        fn = true 1 & pred 0   (missed a real positive)
        tn = true 0 & pred 0   (correct rejection)
    """
    # TODO: make int arrays; count each of the four cases with boolean masks
    raise NotImplementedError


def precision(tp: int, fp: int) -> float:
    """Of everything predicted positive, the fraction that was right: tp / (tp + fp).
    Return 0.0 if nothing was predicted positive."""
    # TODO
    raise NotImplementedError


def recall(tp: int, fn: int) -> float:
    """Of all real positives, the fraction caught: tp / (tp + fn).
    Return 0.0 if there were no real positives."""
    # TODO
    raise NotImplementedError


def f1_score(prec: float, rec: float) -> float:
    """Harmonic mean of precision and recall: 2*p*r / (p + r). Return 0.0 if both are 0."""
    # TODO
    raise NotImplementedError
