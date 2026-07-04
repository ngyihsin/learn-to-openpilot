"""Day 24 homework — Convolutional Neural Networks.

An MLP treats an image as a flat bag of pixels and misses spatial structure. A **CNN** slides
small learnable filters across the image, so it can detect local patterns (edges, textures)
anywhere — the foundation of essentially all vision models, including the one that reads the road
in openpilot.

You'll build a small CNN for 1×8×8 grayscale images and train it to tell "bright row" images from
"bright column" images — a task an MLP struggles with but a conv net nails, because convolution is
built to notice spatial layout.

The classic gotcha: getting the **flattened size** right between the conv/pool stack and the final
Linear layer. Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """1×8×8 -> conv(8 filters) -> ReLU -> maxpool(2) -> flatten -> Linear -> 2 logits."""

    def __init__(self) -> None:
        super().__init__()
        # TODO: define the layers:
        #   self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # keeps 8x8
        #   self.pool = nn.MaxPool2d(2)                             # 8x8 -> 4x4
        #   self.fc   = nn.Linear(8 * 4 * 4, 2)                     # flattened conv output -> 2
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: relu(conv(x)) -> pool -> flatten(start_dim=1) -> fc
        raise NotImplementedError


def build_cnn() -> nn.Module:
    """Return an instance of your CNN."""
    # TODO: return SmallCNN()
    raise NotImplementedError


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 80, lr: float = 0.01) -> list[float]:
    """Standard training loop (same five steps as Day 23), returning per-epoch loss."""
    # TODO: CrossEntropyLoss + Adam; loop zero_grad/forward/loss/backward/step; collect losses
    raise NotImplementedError


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()
