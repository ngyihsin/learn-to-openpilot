"""Day 25 homework — Datasets, DataLoaders & preprocessing.

Real training doesn't shove the whole dataset through the model at once — it streams **mini-
batches**, optionally shuffled each epoch, often from disk. PyTorch standardizes this with two
pieces: a **`Dataset`** (knows its length and how to fetch item i) and a **`DataLoader`** (batches,
shuffles, and can prefetch with worker processes). Getting the input pipeline right is half of
making training fast — a starved GPU/CPU is a common, silent bottleneck.

You'll implement a custom Dataset, wrap it in a DataLoader, and write a standardization transform.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    """Wrap feature tensor X (N×D) and label tensor y (N,) as an indexable dataset."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        # TODO: number of samples
        raise NotImplementedError

    def __getitem__(self, i: int):
        # TODO: return the (features, label) pair at index i
        raise NotImplementedError


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Wrap `dataset` in a DataLoader with the given batch size and shuffle flag."""
    # TODO: return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    raise NotImplementedError


def standardize(X: torch.Tensor) -> torch.Tensor:
    """Return X normalized to zero mean and unit std **per feature (column)**.
    (Standardizing inputs makes training faster and more stable.)"""
    # TODO: subtract the per-column mean, divide by the per-column std (add a tiny epsilon
    #       to avoid divide-by-zero). Use dim=0, keepdim=True.
    raise NotImplementedError
