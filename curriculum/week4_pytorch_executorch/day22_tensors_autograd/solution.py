"""Day 22 reference solution — numerical gradients + a gradient-descent training loop."""
from __future__ import annotations

from typing import Callable

import torch


def numerical_gradient(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    x = x.detach().clone().to(torch.float64)
    grad = torch.zeros_like(x)
    flat = x.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        plus = float(f(x.reshape(x.shape)))
        flat[i] = orig - eps
        minus = float(f(x.reshape(x.shape)))
        flat[i] = orig
        gflat[i] = (plus - minus) / (2 * eps)
    return grad.reshape(x.shape)


def fit_line(
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.05,
    epochs: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    for _ in range(epochs):
        pred = w * x + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.detach(), b.detach()
