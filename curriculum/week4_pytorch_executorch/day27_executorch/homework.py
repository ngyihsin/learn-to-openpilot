"""Day 27 homework — toward on-device inference (ExecuTorch).

Day 26 exported a model to run *somewhere else*. **ExecuTorch** is PyTorch's runtime for running
models on-device — phones, microcontrollers, and exactly the kind of embedded hardware in a comma
device. Its pipeline starts by **lowering** your model to a clean, backend-agnostic graph with
`torch.export`, then compiles that to a `.pte` file the tiny C++ runtime executes with no Python.

The `executorch` package installs separately (see the README), so today's gradeable step is the
**first and most important** one that everything else builds on: lowering an `nn.Module` to an
`ExportedProgram` and running it — verifying it still computes the same thing. That exported graph
is precisely what gets handed to ExecuTorch.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch with `torch.export`.)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def lower_model(model: nn.Module, example_input: torch.Tensor):
    """Lower `model` to an ExportedProgram using torch.export (the entry point to the ExecuTorch
    pipeline). Put the model in eval mode first. Return the ExportedProgram."""
    # TODO:
    #   model.eval()
    #   return torch.export.export(model, (example_input,))
    raise NotImplementedError


def run_exported(exported, x: torch.Tensor) -> torch.Tensor:
    """Run an ExportedProgram on input `x` and return its output.
    (An ExportedProgram gives you a runnable module via `.module()`.)"""
    # TODO: return exported.module()(x)
    raise NotImplementedError
