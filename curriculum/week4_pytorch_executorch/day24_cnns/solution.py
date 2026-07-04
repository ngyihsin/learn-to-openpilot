"""Day 24 reference solution — a small CNN + training loop."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


def build_cnn() -> nn.Module:
    return SmallCNN()


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 80, lr: float = 0.01) -> list[float]:
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
        return (model(X).argmax(dim=1) == y).float().mean().item()
