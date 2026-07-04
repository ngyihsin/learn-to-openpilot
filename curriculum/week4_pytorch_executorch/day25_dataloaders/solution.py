"""Day 25 reference solution — Dataset, DataLoader, standardize."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def standardize(X: torch.Tensor) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    return (X - mean) / (std + 1e-8)
