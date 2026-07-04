"""Auto-grader for Day 28 — quantization.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
Skips if torch isn't installed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


def _load(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load(os.environ.get("LP_IMPL", "homework"))
quantize, model_size_bytes = _impl.quantize, _impl.model_size_bytes


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).eval()


def test_quantized_model_is_smaller():
    model = make_model()
    fp32 = model_size_bytes(model)
    qmodel = quantize(model)
    q = model_size_bytes(qmodel)
    assert q < fp32                       # int8 weights take less room
    assert q < 0.5 * fp32                 # should be a big win (~4x on the Linear weights)


def test_quantized_outputs_stay_close():
    torch.manual_seed(0)
    model = make_model()
    qmodel = quantize(model)
    x = torch.randn(8, 64)
    with torch.no_grad():
        fp_out = model(x)
        q_out = qmodel(x)
    assert q_out.shape == fp_out.shape
    # int8 introduces small error; it should be tiny relative to the signal.
    rel = (q_out - fp_out).abs().mean() / fp_out.abs().mean()
    assert rel < 0.1


def test_quantized_model_still_runs():
    model = make_model()
    qmodel = quantize(model)
    out = qmodel(torch.randn(3, 64))
    assert out.shape == (3, 10)
