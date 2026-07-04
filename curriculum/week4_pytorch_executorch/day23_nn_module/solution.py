"""Day 23 reference solution — MLP classifier + training loop."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_model(in_features: int = 2, hidden: int = 16, out_classes: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_classes),
    )


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 200, lr: float = 0.05) -> list[float]:
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()
