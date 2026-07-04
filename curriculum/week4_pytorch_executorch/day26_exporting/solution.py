"""Day 26 reference solution — TorchScript export & reload."""
from __future__ import annotations

import torch
import torch.nn as nn


def export_model(model: nn.Module, example_input: torch.Tensor, path: str) -> str:
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(path)
    return path


def load_model(path: str):
    return torch.jit.load(path)
