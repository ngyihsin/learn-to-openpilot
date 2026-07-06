"""Day 34 homework — classification: softmax & cross-entropy.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

Regression predicts a number; classification predicts a *category* ("cat/dog/car"). The model outputs
raw scores called **logits**; ``softmax`` turns them into probabilities that sum to 1, and
**cross-entropy** measures how surprised we are by the true answer. These are the exact loss and output
layer of almost every classifier you'll meet — including openpilot's and every detector in Week 7.
"""
from __future__ import annotations

import numpy as np


def softmax(z):
    """Turn a 1-D vector of logits into probabilities (positive, summing to 1).

    Use the numerically STABLE form: subtract the max before exponentiating, so large logits
    don't overflow to inf:  e = exp(z - max(z));  return e / sum(e).
    """
    # TODO
    raise NotImplementedError


def one_hot(label: int, num_classes: int):
    """Return a length-``num_classes`` array of 0.0s with a single 1.0 at index ``label``."""
    # TODO
    raise NotImplementedError


def cross_entropy(probs, label: int) -> float:
    """Cross-entropy loss for ONE example: the negative log-probability the model gave the true class.

    loss = -log(probs[label]).  Clip the probability into [1e-12, 1.0] first so log never sees 0.
    Return a plain float.
    """
    # TODO
    raise NotImplementedError


def predict(probs) -> int:
    """Return the predicted class = index of the largest probability (np.argmax). Return a plain int."""
    # TODO
    raise NotImplementedError


def accuracy(pred_labels, true_labels) -> float:
    """Fraction of predictions that match the truth (0.0–1.0). Return a plain float."""
    # TODO: compare elementwise and take the mean
    raise NotImplementedError
