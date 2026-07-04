"""Day 23 homework — nn.Module & the real training loop.

On Day 22 you wrote gradient descent by hand. PyTorch packages that up: `nn.Module` holds your
parameters and defines `forward`, an **optimizer** does the weight update, and a **loss function**
measures error. The five-step loop is identical to Day 22 — forward, loss, backward, step,
zero_grad — you just don't manage the parameters yourself. Master this and every model in this
week (CNNs, exported models) is a variation on it.

You'll build a small classifier (an MLP), train it on a toy 2-class dataset, and measure accuracy.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_model(in_features: int = 2, hidden: int = 16, out_classes: int = 2) -> nn.Module:
    """Return a small MLP: Linear -> ReLU -> Linear, mapping `in_features` to `out_classes`
    logits. `nn.Sequential` is the easiest way."""
    # TODO: return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_classes))
    raise NotImplementedError


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
          epochs: int = 200, lr: float = 0.05) -> list[float]:
    """Train `model` to classify (X -> y) and return the per-epoch loss.

    Use CrossEntropyLoss (expects raw logits + integer class labels) and an Adam optimizer.
    The loop, exactly as on Day 22:
        opt.zero_grad(); loss = criterion(model(X), y); loss.backward(); opt.step()
    """
    # TODO:
    #   - criterion = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    #   - loop `epochs` times running the 5 steps, appending loss.item() to a history list
    #   - return the history
    raise NotImplementedError


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of examples the model classifies correctly. Use no_grad and argmax over logits."""
    # TODO: with torch.no_grad(): preds = model(X).argmax(dim=1); return (preds == y).float().mean().item()
    raise NotImplementedError
