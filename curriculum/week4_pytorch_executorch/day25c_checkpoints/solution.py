"""Day 25c reference solution — saving/loading checkpoints via state_dict."""
from __future__ import annotations

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["epoch"])


def models_allclose(model_a: torch.nn.Module, model_b: torch.nn.Module) -> bool:
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    if sd_a.keys() != sd_b.keys():
        return False
    return all(torch.allclose(sd_a[k], sd_b[k]) for k in sd_a)
