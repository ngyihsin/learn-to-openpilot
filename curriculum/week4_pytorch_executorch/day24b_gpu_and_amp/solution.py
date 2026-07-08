"""Day 24b reference solution — device management + Automatic Mixed Precision (AMP)."""
from __future__ import annotations

import torch


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(move_to(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: move_to(value, device) for key, value in obj.items()}
    return obj


def count_parameters(model, trainable_only: bool = True) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad or not trainable_only
    )


def autocast_forward(model, x, device):
    with torch.autocast(device_type=device.type), torch.no_grad():
        return model(x)


def amp_train_step(model, x, y, optimizer, loss_fn, device) -> float:
    with torch.autocast(device_type=device.type):
        pred = model(x)
        loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
