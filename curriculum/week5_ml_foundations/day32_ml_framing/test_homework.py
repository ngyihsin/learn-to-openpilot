"""Auto-grader for Day 32 — the ML framing: model, loss, generalization.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings

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
mse = _impl.mse
train_val_split = _impl.train_val_split
fit_polynomial = _impl.fit_polynomial
predict_polynomial = _impl.predict_polynomial
select_degree = _impl.select_degree


def test_mse_zero_when_equal():
    assert mse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_mse_value_and_type():
    # errors of 2 and 2 -> (4 + 4) / 2 = 4.0
    val = mse([1, 2], [3, 4])
    assert val == pytest.approx(4.0)
    assert isinstance(val, float)  # a plain float, not a 0-d array


def test_mse_handles_integer_input():
    # must not do integer division / overflow on int inputs
    assert mse([0, 0], [1, 3]) == pytest.approx(5.0)  # (1 + 9) / 2


def test_split_sizes():
    X = np.arange(100)
    y = np.arange(100) * 2
    Xtr, ytr, Xval, yval = train_val_split(X, y, val_frac=0.2, seed=0)
    assert len(Xval) == 20 and len(yval) == 20
    assert len(Xtr) == 80 and len(ytr) == 80


def test_split_is_partition_and_keeps_pairs():
    X = np.arange(100)
    y = np.arange(100) * 2  # label == 2 * feature, so we can detect a scramble
    Xtr, ytr, Xval, yval = train_val_split(X, y, val_frac=0.2, seed=0)
    # disjoint and covering
    assert set(Xtr.tolist()) & set(Xval.tolist()) == set()
    assert set(Xtr.tolist()) | set(Xval.tolist()) == set(range(100))
    # labels followed their features
    assert np.all(ytr == Xtr * 2)
    assert np.all(yval == Xval * 2)


def test_split_deterministic_with_seed():
    X = np.arange(50)
    y = np.arange(50)
    a = train_val_split(X, y, val_frac=0.3, seed=42)
    b = train_val_split(X, y, val_frac=0.3, seed=42)
    assert np.array_equal(a[2], b[2])  # same validation features
    c = train_val_split(X, y, val_frac=0.3, seed=7)
    assert not np.array_equal(a[2], c[2])  # a different seed shuffles differently


def test_fit_predict_recovers_known_polynomial():
    x = np.linspace(-3, 3, 50)
    y = 2 * x**2 - 3 * x + 1
    coeffs = fit_polynomial(x, y, 2)
    yhat = predict_polynomial(coeffs, x)
    assert mse(y, yhat) < 1e-6
    assert np.allclose(np.asarray(coeffs, dtype=float), [2.0, -3.0, 1.0], atol=1e-6)


def test_select_degree_picks_the_generalizing_model():
    # Data comes from a degree-2 function plus noise. A very high-degree fit will have
    # LOWER training error but HIGHER validation error (overfitting) — so the selector
    # must reject it and land near degree 2.
    rng = np.random.default_rng(0)
    truth = lambda x: 1.5 * x**2 - 0.5 * x + 0.2
    x_tr = np.linspace(-1, 1, 40)
    x_val = np.linspace(-0.95, 0.95, 20)
    y_tr = truth(x_tr) + rng.normal(0, 0.05, size=x_tr.shape)
    y_val = truth(x_val) + rng.normal(0, 0.05, size=x_val.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # high-degree polyfit is poorly conditioned; that's the point
        best = select_degree(x_tr, y_tr, x_val, y_val, degrees=[0, 1, 2, 3, 4, 6, 8])
    assert best in (2, 3, 4)   # near the true complexity
    assert best < 6            # must NOT choose the overfitting high-degree model


def test_select_degree_underfit_is_rejected_too():
    # A pure quadratic (no noise): degree 0 and 1 can't represent it and must lose to degree >= 2.
    x_tr = np.linspace(-2, 2, 30)
    x_val = np.linspace(-1.8, 1.8, 15)
    f = lambda x: x**2
    best = select_degree(x_tr, f(x_tr), x_val, f(x_val), degrees=[0, 1, 2])
    assert best == 2
