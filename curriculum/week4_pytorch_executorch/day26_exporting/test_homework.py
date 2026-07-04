"""Auto-grader for Day 26 — exporting models.

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
export_model, load_model = _impl.export_model, _impl.load_model


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()


def test_export_creates_file(tmp_path):
    model = make_model()
    path = str(tmp_path / "model.pt")
    returned = export_model(model, torch.randn(1, 4), path)
    assert returned == path
    assert os.path.exists(path)


def test_reloaded_model_matches_original(tmp_path):
    model = make_model()
    path = str(tmp_path / "model.pt")
    export_model(model, torch.randn(1, 4), path)
    loaded = load_model(path)

    test_in = torch.randn(5, 4)
    with torch.no_grad():
        original_out = model(test_in)
        loaded_out = loaded(test_in)
    assert torch.allclose(original_out, loaded_out, atol=1e-5)


def test_loaded_is_runnable_without_original_class(tmp_path):
    model = make_model()
    path = str(tmp_path / "m.pt")
    export_model(model, torch.randn(1, 4), path)
    loaded = load_model(path)
    out = loaded(torch.randn(2, 4))
    assert out.shape == (2, 3)
