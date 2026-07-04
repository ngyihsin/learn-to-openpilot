"""Auto-grader for Day 24 — CNNs.

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
build_cnn, train, accuracy = _impl.build_cnn, _impl.train, _impl.accuracy


def make_images(n: int = 120):
    """n grayscale 1x8x8 images: even index = a bright ROW (class 0), odd = a bright COLUMN (1)."""
    torch.manual_seed(0)
    imgs, labels = [], []
    for i in range(n):
        img = torch.rand(1, 8, 8) * 0.1
        if i % 2 == 0:
            img[0, torch.randint(0, 8, (1,)).item(), :] += 1.0
            labels.append(0)
        else:
            img[0, :, torch.randint(0, 8, (1,)).item()] += 1.0
            labels.append(1)
        imgs.append(img)
    return torch.stack(imgs), torch.tensor(labels)


def test_forward_shape():
    model = build_cnn()
    assert isinstance(model, nn.Module)
    out = model(torch.randn(4, 1, 8, 8))
    assert out.shape == (4, 2)


def test_training_reduces_loss():
    torch.manual_seed(0)
    X, y = make_images()
    model = build_cnn()
    history = train(model, X, y, epochs=80, lr=0.01)
    assert history[-1] < history[0]


def test_cnn_learns_rows_vs_columns():
    torch.manual_seed(0)
    X, y = make_images()
    model = build_cnn()
    train(model, X, y, epochs=80, lr=0.01)
    assert accuracy(model, X, y) > 0.9
