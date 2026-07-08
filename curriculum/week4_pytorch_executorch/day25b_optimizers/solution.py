"""Day 25b reference solution — optimizers + learning-rate schedules."""
from __future__ import annotations

import warnings

import torch


def make_optimizer(
    model: torch.nn.Module,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
):
    key = name.lower()
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name!r}")


def step_decay_lr(base_lr: float, step: int, drop: float = 0.5, every: int = 10) -> float:
    return base_lr * (drop ** (step // every))


def warmup_lr(base_lr: float, step: int, warmup_steps: int) -> float:
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def lr_schedule(optimizer, scheduler, n_steps: int) -> list:
    lrs = []
    # We only want to *read* the LR the scheduler produces each step. In real training
    # you call optimizer.step() before scheduler.step(); here there's no optimizer.step(),
    # so PyTorch would warn about the order — silence just that cosmetic warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step.*")
        for _ in range(n_steps):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
    return lrs
