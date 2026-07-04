"""Day 28 reference solution — dynamic quantization + size measurement."""
from __future__ import annotations

import io

import torch
import torch.nn as nn


def quantize(model: nn.Module) -> nn.Module:
    model.eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def model_size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
