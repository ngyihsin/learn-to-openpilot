"""Auto-grader for Day 23 — nn.Module & training loop.

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
build_model, train, accuracy = _impl.build_model, _impl.train, _impl.accuracy


def make_data(n: int = 100):
    torch.manual_seed(0)
    c0 = torch.randn(n, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    c1 = torch.randn(n, 2) * 0.5 + torch.tensor([2.0, 2.0])
    X = torch.cat([c0, c1])
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return X, y


def test_model_shape():
    model = build_model(2, 16, 2)
    assert isinstance(model, nn.Module)
    out = model(torch.randn(5, 2))
    assert out.shape == (5, 2)


def test_training_reduces_loss():
    torch.manual_seed(0)
    X, y = make_data()
    model = build_model()
    history = train(model, X, y, epochs=150, lr=0.05)
    assert len(history) == 150
    assert history[-1] < history[0]              # loss went down


def test_learns_to_classify():
    torch.manual_seed(0)
    X, y = make_data()
    model = build_model()
    train(model, X, y, epochs=150, lr=0.05)
    acc = accuracy(model, X, y)
    assert acc > 0.95                            # two well-separated blobs -> near-perfect


def test_accuracy_range():
    model = build_model()
    X, y = make_data(20)
    acc = accuracy(model, X, y)
    assert 0.0 <= acc <= 1.0
