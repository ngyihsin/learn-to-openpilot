"""Auto-grader for Day 33 — regression by gradient descent.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest


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
predict = _impl.predict
mse_loss = _impl.mse_loss
gradients = _impl.gradients
gradient_descent_step = _impl.gradient_descent_step
fit_linear = _impl.fit_linear


def test_predict_is_line():
    x = np.array([0.0, 1.0, 2.0])
    assert np.allclose(predict(2.0, 1.0, x), [1.0, 3.0, 5.0])


def test_mse_loss_value_and_type():
    x = np.array([1.0, 2.0])
    y = np.array([0.0, 0.0])
    # pred = [1, 2]; squared errors [1, 4]; mean = 2.5
    val = mse_loss(1.0, 0.0, x, y)
    assert val == pytest.approx(2.5)
    assert isinstance(val, float)


def test_gradients_match_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    y = rng.normal(size=20)
    w, b, eps = 0.3, -0.4, 1e-6
    dw, db = gradients(w, b, x, y)
    num_dw = (mse_loss(w + eps, b, x, y) - mse_loss(w - eps, b, x, y)) / (2 * eps)
    num_db = (mse_loss(w, b + eps, x, y) - mse_loss(w, b - eps, x, y)) / (2 * eps)
    assert dw == pytest.approx(num_dw, abs=1e-4)
    assert db == pytest.approx(num_db, abs=1e-4)


def test_one_step_decreases_loss():
    x = np.linspace(-2, 2, 30)
    y = 3 * x + 2
    before = mse_loss(0.0, 0.0, x, y)
    w1, b1 = gradient_descent_step(0.0, 0.0, x, y, lr=0.05)
    after = mse_loss(w1, b1, x, y)
    assert after < before


def test_step_moves_in_right_direction():
    # data slopes up (w should rise from 0) and is shifted up (b should rise from 0)
    x = np.linspace(-2, 2, 30)
    y = 3 * x + 2
    w1, b1 = gradient_descent_step(0.0, 0.0, x, y, lr=0.05)
    assert w1 > 0
    assert b1 > 0


def test_fit_recovers_line():
    x = np.linspace(-2, 2, 50)
    y = 3 * x + 2
    w, b = fit_linear(x, y, lr=0.05, epochs=3000)
    assert w == pytest.approx(3.0, abs=1e-2)
    assert b == pytest.approx(2.0, abs=1e-2)


def test_fit_is_deterministic():
    x = np.linspace(0, 1, 25)
    y = -1.5 * x + 0.5
    a = fit_linear(x, y, lr=0.1, epochs=500)
    b = fit_linear(x, y, lr=0.1, epochs=500)
    assert a == b  # same start (0,0), same data, same steps -> identical result
