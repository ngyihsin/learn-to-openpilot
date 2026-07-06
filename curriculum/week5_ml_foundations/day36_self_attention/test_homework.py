"""Auto-grader for Day 36 — scaled dot-product self-attention.

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
softmax = _impl.softmax
scaled_dot_product_attention = _impl.scaled_dot_product_attention
self_attention = _impl.self_attention


def test_softmax_rows_sum_to_one():
    p = softmax([[0.0, 0.0], [1.0, 3.0]], axis=-1)
    assert np.allclose(p.sum(axis=-1), [1.0, 1.0])
    assert np.allclose(p[0], [0.5, 0.5])
    assert p[1, 1] > p[1, 0]                       # bigger logit -> bigger weight


def test_softmax_stable():
    p = softmax([[1000.0, 1000.0]], axis=-1)
    assert np.all(np.isfinite(p))
    assert np.allclose(p, [[0.5, 0.5]])


def test_attention_weights_are_distributions():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 2))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert w.shape == (3, 5)                        # one weight per (query, key)
    assert out.shape == (3, 2)                      # one output row per query, in V's width
    assert np.allclose(w.sum(axis=-1), np.ones(3))  # each query's weights sum to 1


def test_identical_keys_give_uniform_weights_and_mean_value():
    # all keys equal -> every query matches them equally -> output is the mean of V
    Q = [[1.0, 2.0]]
    K = [[0.0, 0.0], [0.0, 0.0]]
    V = [[1.0, 1.0], [3.0, 3.0]]
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w, [[0.5, 0.5]])
    assert np.allclose(out, [[2.0, 2.0]])


def test_dominant_key_wins():
    # query aligns strongly with key 0 -> nearly all weight there -> output ~ value 0
    Q = [[10.0, 0.0]]
    K = [[10.0, 0.0], [0.0, 10.0]]
    V = [[5.0, 6.0], [7.0, 8.0]]
    out, w = scaled_dot_product_attention(Q, K, V)
    assert w[0, 0] > 0.99
    assert np.allclose(out, [[5.0, 6.0]], atol=1e-2)


def test_scaling_by_sqrt_dk():
    # scores must be divided by sqrt(d_k); check the pre-softmax scaling via a crafted case.
    # Q=K=[[1,1,1,1]] -> raw dot = 4; /sqrt(4)=2. Single key -> weight 1, output = V.
    Q = [[1.0, 1.0, 1.0, 1.0]]
    K = [[1.0, 1.0, 1.0, 1.0]]
    V = [[9.0]]
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w, [[1.0]])
    assert np.allclose(out, [[9.0]])


def test_self_attention_reduces_to_attention_with_identity_projections():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    I = np.eye(2)
    o1, w1 = self_attention(X, I, I, I)
    o2, w2 = scaled_dot_product_attention(X, X, X)
    assert np.allclose(o1, o2)
    assert np.allclose(w1, w2)
    assert o1.shape == (3, 2)
