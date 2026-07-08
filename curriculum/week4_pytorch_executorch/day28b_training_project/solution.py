"""Day 28b reference solution — a full end-to-end training project.

Dataset -> train/val split -> training loop with per-epoch validation -> best checkpoint.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_dataset(n_samples: int = 512, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0
    x0 = torch.randn(n0, 2, generator=g) + torch.tensor([-2.0, -2.0])
    x1 = torch.randn(n1, 2, generator=g) + torch.tensor([2.0, 2.0])
    X = torch.cat([x0, x1], dim=0).to(torch.float32)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).to(torch.long)
    return TensorDataset(X, y)


def split_dataset(ds: TensorDataset, val_frac: float = 0.2, seed: int = 0):
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=g)


def build_model(in_dim: int = 2, hidden: int = 16, n_classes: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes),
    )


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            preds = model(X).argmax(dim=1)
            correct += int((preds == y).sum())
            total += y.numel()
    return correct / total if total else 0.0


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.1,
) -> dict:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loss: list[float] = []
    val_acc: list[float] = []
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for X, y in train_loader:
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        mean_loss = running / n_batches if n_batches else 0.0
        train_loss.append(mean_loss)

        acc = evaluate(model, val_loader)
        val_acc.append(acc)
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    return {
        "train_loss": train_loss,
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "best_state": best_state,
    }
