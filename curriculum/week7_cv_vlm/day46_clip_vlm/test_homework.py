"""Auto-grader for Day 46 — CLIP-style image/text matching.

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
l2_normalize = _impl.l2_normalize
cosine_similarity = _impl.cosine_similarity
similarity_matrix = _impl.similarity_matrix
classify = _impl.classify


def test_l2_normalize():
    assert np.allclose(l2_normalize([3.0, 4.0]), [0.6, 0.8])
    assert np.linalg.norm(l2_normalize([2.0, 5.0, 1.0])) == pytest.approx(1.0)


def test_l2_normalize_handles_zero_vector():
    assert np.allclose(l2_normalize([0.0, 0.0]), [0.0, 0.0])   # no NaN from dividing by zero


def test_l2_normalize_rows_of_matrix():
    out = l2_normalize([[3.0, 4.0], [0.0, 5.0]])
    assert np.allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])


def test_cosine_similarity_extremes():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)   # identical
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)         # orthogonal
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)       # opposite


def test_similarity_matrix():
    imgs = np.array([[1.0, 0.0], [0.0, 1.0]])
    txts = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    S = similarity_matrix(imgs, txts)
    assert S.shape == (2, 3)
    assert np.allclose(S, [[1.0, 0.0, 0.70710678], [0.0, 1.0, 0.70710678]])


def test_classify_picks_nearest_text():
    txts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert classify([0.9, 0.1], txts) == 0     # closest to text 0
    assert classify([0.1, 0.9], txts) == 1     # closest to text 1
    assert isinstance(classify([1.0, 0.0], txts), int)
