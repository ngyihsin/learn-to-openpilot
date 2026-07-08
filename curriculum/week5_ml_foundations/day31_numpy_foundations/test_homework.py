"""Auto-grader for Day 31 — numpy & array thinking.

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
scale_and_shift = _impl.scale_and_shift
normalize = _impl.normalize
select_positive = _impl.select_positive
row_means = _impl.row_means
closest_index = _impl.closest_index
reproducible_randoms = _impl.reproducible_randoms
reorder = _impl.reorder
matmul = _impl.matmul


def test_scale_and_shift_is_vectorized():
    out = scale_and_shift([0, 1, 2], 2.0, 1.0)
    assert isinstance(out, np.ndarray)   # a whole array back, not a Python list
    assert np.allclose(out, [1.0, 3.0, 5.0])


def test_normalize_gives_mean0_std1():
    out = normalize([1.0, 2.0, 3.0])
    assert np.allclose(out, [-1.22474487, 0.0, 1.22474487])
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0, abs=1e-9)


def test_select_positive_masks_correctly():
    out = select_positive([-1.0, 2.0, -3.0, 4.0, 0.0])
    assert np.allclose(out, [2.0, 4.0])   # 0 is NOT strictly positive


def test_row_means_uses_the_right_axis():
    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out = row_means(M)
    assert out.shape == (2,)              # one mean per row, not per column
    assert np.allclose(out, [2.0, 5.0])


def test_closest_index():
    assert closest_index([10.0, 20.0, 30.0], 22.0) == 1
    assert closest_index([10.0, 20.0, 30.0], 9.0) == 0
    assert isinstance(closest_index([1.0, 2.0], 1.4), int)


def test_reproducible_randoms_is_seeded():
    a = reproducible_randoms(5, seed=0)
    b = reproducible_randoms(5, seed=0)
    assert a.shape == (5,)
    assert np.array_equal(a, b)           # same seed -> identical draw
    assert np.all((a >= 0) & (a < 1))     # floats in [0, 1)
    c = reproducible_randoms(5, seed=1)
    assert not np.array_equal(a, c)       # a different seed differs


def test_reorder_by_position():
    v = np.array([5.0, -2.0, 9.0, -1.0, 4.0])
    assert np.allclose(reorder(v, [2, 0, 4]), [9.0, 5.0, 4.0])   # pick positions 2,0,4
    assert np.allclose(reorder([10.0, 20.0, 30.0], [2, 1, 0]), [30.0, 20.0, 10.0])  # reverse as a shuffle


def test_matmul_values_and_shape():
    out = matmul([[1, 2], [3, 4]], [[5], [6]])          # (2,2) @ (2,1) -> (2,1)
    assert out.shape == (2, 1)
    assert np.allclose(out, [[17.0], [39.0]])           # row·column: 1*5+2*6, 3*5+4*6


def test_matmul_with_transpose():
    A = np.array([[1.0, 2.0, 3.0]])                     # (1,3)
    # A @ A fails (inner sizes 3 vs 1) — transposing fixes the shapes:
    assert np.allclose(matmul(A, A.T), [[14.0]])        # (1,3) @ (3,1) -> (1,1): 1+4+9
