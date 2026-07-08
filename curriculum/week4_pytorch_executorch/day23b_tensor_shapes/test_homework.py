"""Auto-grader for Day 23b — Tensor Shapes & Broadcasting.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

Skips automatically if torch isn't installed (it's only needed for Week 4).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")  # Week-4 only dependency


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
broadcast_shapes = _impl.broadcast_shapes
batched_linear = _impl.batched_linear
to_shape = _impl.to_shape
channel_normalize = _impl.channel_normalize


@pytest.mark.parametrize(
    "a, b",
    [
        ((3, 1), (1, 4)),
        ((2, 1, 4), (5, 4)),
        ((), (3,)),
        ((1,), (3, 3)),
    ],
)
def test_broadcast_shapes_matches_torch(a, b):
    expected = tuple(torch.broadcast_shapes(a, b))
    assert tuple(broadcast_shapes(a, b)) == expected


def test_broadcast_shapes_incompatible_raises():
    with pytest.raises(ValueError):
        broadcast_shapes((3,), (4,))


def test_batched_linear_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(8, 5)
    W = torch.randn(3, 5)   # (out, in)
    b = torch.randn(3)
    out = batched_linear(x, W, b)
    assert out.shape == (8, 3)
    assert torch.allclose(out, x @ W.T + b, atol=1e-6)


def test_to_shape_reshapes():
    x = torch.arange(12)
    out = to_shape(x, (3, 4))
    assert out.shape == (3, 4)
    assert torch.equal(out.reshape(-1), x)


def test_to_shape_bad_size_raises():
    x = torch.arange(12)
    with pytest.raises(ValueError):
        to_shape(x, (5, 5))


def test_channel_normalize_stats():
    torch.manual_seed(0)
    img = torch.randn(3, 4, 4)
    out = channel_normalize(img)
    assert out.shape == img.shape
    mean = out.mean(dim=(1, 2))
    std = out.std(dim=(1, 2))
    assert torch.allclose(mean, torch.zeros(3), atol=1e-5)
    assert torch.allclose(std, torch.ones(3), atol=1e-4)
