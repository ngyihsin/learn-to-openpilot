"""Auto-grader for Day 22 — Tensors & Autograd.

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
numerical_gradient = _impl.numerical_gradient
fit_line = _impl.fit_line


def test_numerical_gradient_of_sum_of_squares():
    # f(x) = sum(x^2)  ->  grad = 2x
    x = torch.tensor([1.0, -2.0, 3.0])
    g = numerical_gradient(lambda t: (t ** 2).sum(), x)
    assert torch.allclose(g.to(torch.float32), 2 * x, atol=1e-2)


def test_numerical_gradient_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(5, dtype=torch.float64)

    def f(t):
        return (torch.sin(t) * t).sum()

    num = numerical_gradient(f, x)

    xa = x.clone().requires_grad_(True)
    f(xa).backward()
    assert torch.allclose(num, xa.grad, atol=1e-4)


def test_fit_line_recovers_parameters():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 100)
    y = 3.0 * x + 2.0                       # clean line: slope 3, intercept 2
    w, b = fit_line(x, y, lr=0.1, epochs=2000)
    assert float(w) == pytest.approx(3.0, abs=0.05)
    assert float(b) == pytest.approx(2.0, abs=0.05)


def test_fit_line_reduces_loss():
    torch.manual_seed(1)
    x = torch.linspace(-1, 1, 50)
    y = -2.0 * x + 0.5 + 0.01 * torch.randn(50)

    def mse(w, b):
        return float(((w * x + b - y) ** 2).mean())

    w, b = fit_line(x, y, lr=0.1, epochs=1000)
    assert mse(w, b) < mse(torch.zeros(()), torch.zeros(()))  # better than the w=b=0 start
    assert mse(w, b) < 0.05
