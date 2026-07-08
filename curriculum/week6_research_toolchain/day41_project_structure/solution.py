"""Day 41 reference solution — reproducibility utilities for a PyTorch research project.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.
Pure Python + numpy (no torch needed) so it runs anywhere.
"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every source of randomness so a run is reproducible.
    (In a real project you'd also seed torch here — guarded so this file needs no torch.)"""
    random.seed(seed)
    np.random.seed(seed)
    try:  # seed torch too if it happens to be installed
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


class AverageMeter:
    """Track a running average of a metric across a training loop (loss, accuracy, ...)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def merge_config(defaults: dict, overrides: dict) -> dict:
    """Return a NEW config = defaults with overrides applied on top. Must not mutate either input."""
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def pick_best(history: list[dict], key: str, mode: str = "min") -> dict:
    """Return the epoch-record (a dict) with the best ``key`` value.
    ``mode='min'`` for losses, ``mode='max'`` for accuracy. Raise ValueError on empty/bad mode."""
    if not history:
        raise ValueError("history is empty")
    if mode == "min":
        return min(history, key=lambda h: h[key])
    if mode == "max":
        return max(history, key=lambda h: h[key])
    raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
