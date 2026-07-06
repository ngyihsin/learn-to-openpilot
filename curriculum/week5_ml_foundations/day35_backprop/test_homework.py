"""Auto-grader for Day 35 — a 2-layer neural network with backpropagation.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

The key test is ``test_backward_matches_numerical_gradient``: it checks your hand-derived gradients
against a finite-difference estimate. If backprop is right, they agree to ~1e-5.
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
relu = _impl.relu
relu_grad = _impl.relu_grad
forward = _impl.forward
mse_loss = _impl.mse_loss
backward = _impl.backward
train = _impl.train


def test_relu_and_grad():
    assert np.allclose(relu([-1.0, 0.0, 2.0]), [0.0, 0.0, 2.0])
    assert np.allclose(relu_grad([-1.0, 0.0, 2.0]), [0.0, 0.0, 1.0])  # 0 is not > 0


def test_forward_exact_small_case():
    X = np.array([[1.0, 2.0]])
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])   # identity -> z1 = X
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[1.0], [1.0]])             # sum the two hidden units
    b2 = np.array([0.0])
    y, z1, h = forward(X, W1, b1, W2, b2)
    assert np.allclose(z1, [[1.0, 2.0]])
    assert np.allclose(h, [[1.0, 2.0]])
    assert np.allclose(y, [[3.0]])


def test_forward_relu_clips_negatives():
    X = np.array([[-1.0, 2.0]])
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.array([[1.0], [1.0]])
    b2 = np.array([0.0])
    y, z1, h = forward(X, W1, b1, W2, b2)
    assert np.allclose(h, [[0.0, 2.0]])       # the -1 got clipped to 0
    assert np.allclose(y, [[2.0]])


def test_mse_loss():
    assert mse_loss([[1.0], [2.0]], [[1.0], [0.0]]) == pytest.approx(2.0)  # mean(0, 4)


def _numerical_grad(f, P, eps=1e-6):
    g = np.zeros_like(P, dtype=float)
    it = np.nditer(P, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = P[idx]
        P[idx] = old + eps
        lp = f()
        P[idx] = old - eps
        lm = f()
        P[idx] = old
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return g


def test_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    N, Din, H, Dout = 5, 3, 4, 2
    X = rng.normal(size=(N, Din))
    T = rng.normal(size=(N, Dout))
    W1 = rng.normal(size=(Din, H))
    b1 = rng.normal(size=(H,))
    W2 = rng.normal(size=(H, Dout))
    b2 = rng.normal(size=(Dout,))

    y, z1, h = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(X, z1, h, W2, y, T)

    loss = lambda: mse_loss(forward(X, W1, b1, W2, b2)[0], T)
    assert np.allclose(dW1, _numerical_grad(loss, W1), atol=1e-5)
    assert np.allclose(db1, _numerical_grad(loss, b1), atol=1e-5)
    assert np.allclose(dW2, _numerical_grad(loss, W2), atol=1e-5)
    assert np.allclose(db2, _numerical_grad(loss, b2), atol=1e-5)


def test_train_reduces_loss():
    rng = np.random.default_rng(1)
    N, Din, H, Dout = 8, 3, 5, 1
    X = rng.normal(size=(N, Din))
    T = rng.normal(size=(N, Dout))
    W1 = rng.normal(size=(Din, H)) * 0.5
    b1 = np.zeros(H)
    W2 = rng.normal(size=(H, Dout)) * 0.5
    b2 = np.zeros(Dout)

    before = mse_loss(forward(X, W1, b1, W2, b2)[0], T)
    W1n, b1n, W2n, b2n = train(X, T, W1, b1, W2, b2, lr=0.05, epochs=300)
    after = mse_loss(forward(X, W1n, b1n, W2n, b2n)[0], T)
    assert after < before * 0.5          # the network clearly learned
    # train must not mutate the caller's original arrays in place
    assert mse_loss(forward(X, W1, b1, W2, b2)[0], T) == pytest.approx(before)
