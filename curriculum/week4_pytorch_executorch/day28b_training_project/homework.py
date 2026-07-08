"""Day 28b homework — A Full Training Project (End-to-End).

This is the Week 4 capstone: everything you've built this week, wired into one real
experiment. You'll construct a **Dataset**, split it into **train/val**, write a training
loop with **per-epoch validation**, keep the **best checkpoint**, and report **accuracy**.

Nothing here is toy magic — this exact skeleton (dataset → split → loop → validate →
checkpoint) is what every real experiment you'll run looks like, from a tiny MLP to
openpilot's driving model.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_dataset(n_samples: int = 512, seed: int = 0) -> TensorDataset:
    """Build a two-class 2D toy dataset ("blobs") wrapped in a ``TensorDataset``.

    Class 0 is drawn from a Normal centered at (-2, -2); class 1 from a Normal centered
    at (+2, +2); both with std ~1.0. Roughly half the points are class 0, half class 1.
    The two clouds barely overlap, so a good model should reach >90% accuracy.

    Return a ``TensorDataset(X, y)`` where:
        - ``X`` is float32, shape ``(n_samples, 2)``
        - ``y`` is long (int64), shape ``(n_samples,)``, values in {0, 1}

    Seed a **local** ``torch.Generator().manual_seed(seed)`` and pass it to every random
    call so the dataset is reproducible without touching the global RNG.
    """
    # TODO:
    #   - g = torch.Generator().manual_seed(seed)
    #   - split n_samples into n0 (class 0) and n1 (class 1), e.g. n0 = n_samples // 2
    #   - X0 = torch.randn(n0, 2, generator=g) + centre0 ; X1 = ... + centre1
    #   - build y (n0 zeros, n1 ones) as long, stack X as float32, wrap in TensorDataset
    raise NotImplementedError


def split_dataset(ds: TensorDataset, val_frac: float = 0.2, seed: int = 0):
    """Split ``ds`` into ``(train_ds, val_ds)`` using ``random_split``.

    ``val_ds`` gets ``floor(val_frac * len(ds))`` samples; ``train_ds`` gets the rest, so
    the two sizes always sum to ``len(ds)``. Use a seeded generator so the split is
    reproducible.
    """
    # TODO:
    #   - n_val = int(len(ds) * val_frac) ; n_train = len(ds) - n_val
    #   - g = torch.Generator().manual_seed(seed)
    #   - return random_split(ds, [n_train, n_val], generator=g)
    raise NotImplementedError


def build_model(in_dim: int = 2, hidden: int = 16, n_classes: int = 2) -> nn.Module:
    """Return a small MLP classifier: ``Linear(in_dim, hidden) -> ReLU -> Linear(hidden, n_classes)``.

    It outputs raw **logits** (one per class) — no softmax. ``CrossEntropyLoss`` expects
    logits, and ``argmax`` over logits gives the predicted class.
    """
    # TODO:
    #   - build and return an nn.Sequential (or a custom nn.Module) with the layers above
    raise NotImplementedError


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Return classification accuracy in [0, 1] over every batch in ``loader``.

    Put the model in eval mode, run under ``torch.no_grad()`` (no gradients needed for
    scoring), take ``argmax`` over the logits, and divide correct predictions by total.
    """
    # TODO:
    #   - model.eval(); with torch.no_grad(): loop over (X, y) batches
    #   - preds = model(X).argmax(dim=1) ; count preds == y
    #   - return correct / total  (guard against an empty loader if you like)
    raise NotImplementedError


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.1,
) -> dict:
    """Train ``model`` and return a history dict.

    Use ``SGD(model.parameters(), lr=lr)`` and ``CrossEntropyLoss``. Each epoch:
        1. one training pass over ``train_loader`` (forward → loss → backward → step →
           zero_grad), tracking the **mean** training loss for the epoch;
        2. call ``evaluate(model, val_loader)`` to get validation accuracy.

    Keep the **best** model seen so far: whenever val accuracy improves, record it and save
    a ``copy.deepcopy`` of ``model.state_dict()`` (deepcopy so later training can't mutate
    the saved weights).

    Return:
        {
            "train_loss":   [mean loss per epoch],   # length == epochs
            "val_acc":      [val accuracy per epoch], # length == epochs
            "best_val_acc": float,                    # max of val_acc
            "best_state":   state_dict at the best epoch,
        }
    """
    # TODO:
    #   - opt = torch.optim.SGD(model.parameters(), lr=lr) ; loss_fn = nn.CrossEntropyLoss()
    #   - for each epoch: sum batch losses, average them, then acc = evaluate(...)
    #   - track best_val_acc and copy.deepcopy(model.state_dict()) when it improves
    #   - return the history dict described above
    raise NotImplementedError
