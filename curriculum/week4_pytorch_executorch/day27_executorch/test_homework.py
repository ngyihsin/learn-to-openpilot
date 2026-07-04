"""Auto-grader for Day 27 — lowering toward ExecuTorch.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
Skips if torch (with torch.export) isn't installed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "export"):
    pytest.skip("this torch build lacks torch.export", allow_module_level=True)
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
lower_model, run_exported = _impl.lower_model, _impl.run_exported


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()


def test_lowering_produces_exported_program():
    model = make_model()
    ep = lower_model(model, torch.randn(3, 4))
    assert isinstance(ep, torch.export.ExportedProgram)


def test_exported_matches_eager():
    model = make_model()
    example = torch.randn(3, 4)
    ep = lower_model(model, example)

    test_in = torch.randn(3, 4)
    with torch.no_grad():
        eager = model(test_in)
    exported = run_exported(ep, test_in)
    assert exported.shape == (3, 3)
    assert torch.allclose(eager, exported, atol=1e-5)
