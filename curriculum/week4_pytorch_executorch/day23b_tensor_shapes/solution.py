"""Day 23b reference solution — tensor shapes & broadcasting."""
from __future__ import annotations

import torch


def broadcast_shapes(a: tuple, b: tuple) -> tuple:
    a = tuple(a)
    b = tuple(b)
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + a  # pad on the left with 1s
    b = (1,) * (n - len(b)) + b
    out = []
    for da, db in zip(a, b):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(f"shapes {a} and {b} are not broadcastable at dims {da} vs {db}")
    return tuple(out)


def batched_linear(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x @ W.T + b


def to_shape(x: torch.Tensor, shape: tuple) -> torch.Tensor:
    target = 1
    for s in shape:
        target *= s
    if target != x.numel():
        raise ValueError(
            f"cannot reshape tensor of {x.numel()} elements into shape {tuple(shape)} "
            f"({target} elements)"
        )
    return x.reshape(shape)


def channel_normalize(img: torch.Tensor) -> torch.Tensor:
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True)
    return (img - mean) / (std + 1e-8)
