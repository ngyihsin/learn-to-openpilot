"""Day 28c reference solution — steering-angle regression."""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 16


def make_road_dataset(n: int = 512, seed: int = 0) -> TensorDataset:
    """PROVIDED — identical to homework.py."""
    g = torch.Generator().manual_seed(seed)
    cols = torch.randint(2, IMG - 3, (n,), generator=g)
    x = 0.05 * torch.randn(n, 1, IMG, IMG, generator=g)
    for i, c in enumerate(cols):
        x[i, 0, :, c : c + 2] = 1.0
    center = (IMG - 1) / 2
    y = (cols.float() - center) / center
    return TensorDataset(x, y)


def build_regressor() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def mae(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).squeeze(-1)
            total += (pred - yb).abs().sum().item()
            count += yb.numel()
    return total / count


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 12,
    lr: float = 1e-2,
) -> tuple[dict, float]:
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_mae = copy.deepcopy(model.state_dict()), float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        val = mae(model, val_loader)
        if val < best_mae:
            best_mae = val
            best_state = copy.deepcopy(model.state_dict())
    return best_state, best_mae


def measure_latency_ms(
    model: nn.Module, example: torch.Tensor, warmup: int = 5, iters: int = 50
) -> float:
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(example)
        for _ in range(iters):
            t0 = time.perf_counter()
            model(example)
            times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    median = times[n // 2] if n % 2 else (times[n // 2 - 1] + times[n // 2]) / 2
    return median * 1000.0


def meets_frame_budget(latency_ms: float, fps: float) -> bool:
    return latency_ms <= 1000.0 / fps
