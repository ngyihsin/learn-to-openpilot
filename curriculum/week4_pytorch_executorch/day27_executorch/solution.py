"""Day 27 reference solution — lowering with torch.export (the ExecuTorch entry point)."""
from __future__ import annotations

import torch
import torch.nn as nn


def lower_model(model: nn.Module, example_input: torch.Tensor):
    model.eval()
    return torch.export.export(model, (example_input,))


def run_exported(exported, x: torch.Tensor) -> torch.Tensor:
    return exported.module()(x)
