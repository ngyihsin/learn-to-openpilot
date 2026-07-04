"""Day 28 homework — quantization: making models small and fast.

A trained model stores its weights as 32-bit floats. On-device, that's often too big and too slow.
**Quantization** stores weights (and sometimes activations) as 8-bit integers instead — roughly
**4× smaller** and faster to compute, for a small, usually-acceptable loss in accuracy. It's a
standard final step before shipping a model to constrained hardware like the comma device.

The simplest flavor is **dynamic quantization**: it quantizes the weights of `Linear` layers to
int8 and quantizes activations on the fly at inference. You'll apply it and confirm two things: the
model gets smaller, and its outputs stay close to the float version.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import io

import torch
import torch.nn as nn


def quantize(model: nn.Module) -> nn.Module:
    """Return a dynamically-quantized copy of `model` (Linear layers -> int8).
    Put the model in eval mode first."""
    # TODO:
    #   model.eval()
    #   return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    raise NotImplementedError


def model_size_bytes(model: nn.Module) -> int:
    """Serialize the model's state_dict to a bytes buffer and return its size in bytes.
    (A fair way to compare on-disk footprint before vs. after quantization.)"""
    # TODO:
    #   buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getbuffer().nbytes
    raise NotImplementedError
