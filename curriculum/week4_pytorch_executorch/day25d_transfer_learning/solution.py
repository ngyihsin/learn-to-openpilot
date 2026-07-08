"""Day 25d reference solution — freeze / unfreeze / swap-head / fine-tune primitives."""
from __future__ import annotations

import torch


def freeze(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def replace_head(model: torch.nn.Module, new_head: torch.nn.Module) -> None:
    model.head = new_head


def trainable_param_names(model: torch.nn.Module) -> list:
    return sorted(name for name, p in model.named_parameters() if p.requires_grad)


def finetune_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
) -> float:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()
