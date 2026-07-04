"""Day 26 homework — exporting a trained model.

A model you train in Python is a live `nn.Module` full of Python code. To *ship* it — to run it
in a C++ app, a mobile runtime, or on the comma device — you **export** it into a portable,
self-contained form that doesn't need your training script or even Python. **TorchScript**
(`torch.jit`) is the classic route: `trace` runs your model once to record the operations, then
`save` writes a `.pt` file you can reload anywhere.

The golden rule of exporting: the reloaded model must produce the **same outputs** as the original.
You'll export, reload, and verify exactly that.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def export_model(model: nn.Module, example_input: torch.Tensor, path: str) -> str:
    """Trace `model` with `example_input` and save the TorchScript module to `path`.
    Return the path. (Put the model in eval mode first so layers like dropout behave for inference.)"""
    # TODO:
    #   model.eval()
    #   traced = torch.jit.trace(model, example_input)
    #   traced.save(path)
    #   return path
    raise NotImplementedError


def load_model(path: str):
    """Load a TorchScript module previously saved to `path` (no original class needed)."""
    # TODO: return torch.jit.load(path)
    raise NotImplementedError
