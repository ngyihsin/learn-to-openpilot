"""Auto-grader for Day 34 — classification: softmax & cross-entropy.

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
one_hot = _impl.one_hot
cross_entropy = _impl.cross_entropy
predict = _impl.predict
accuracy = _impl.accuracy


def test_softmax_is_a_distribution():
    p = softmax([0.0, 0.0])
    assert np.allclose(p, [0.5, 0.5])
    p3 = softmax([1.0, 2.0, 3.0])
    assert p3.sum() == pytest.approx(1.0)     # sums to 1
    assert np.all(p3 > 0)                      # all positive
    assert p3[2] > p3[1] > p3[0]               # bigger logit -> bigger probability


def test_softmax_is_numerically_stable():
    # naive exp(1000) overflows to inf -> nan; the stable form must return [0.5, 0.5]
    p = softmax([1000.0, 1000.0])
    assert np.all(np.isfinite(p))
    assert np.allclose(p, [0.5, 0.5])


def test_one_hot():
    assert np.allclose(one_hot(2, 4), [0.0, 0.0, 1.0, 0.0])
    assert one_hot(0, 3).sum() == pytest.approx(1.0)


def test_cross_entropy_values():
    # uniform 2-class guess -> -log(0.5) = ln 2
    assert cross_entropy([0.5, 0.5], 0) == pytest.approx(np.log(2))
    # confident and correct -> tiny loss
    assert cross_entropy([0.01, 0.99], 1) < 0.02
    # confident and WRONG -> big loss
    assert cross_entropy([0.01, 0.99], 0) > 4.0
    assert isinstance(cross_entropy([0.5, 0.5], 0), float)


def test_cross_entropy_handles_zero_prob():
    # true class got probability 0 -> must not be inf/nan (clipping), just large
    loss = cross_entropy([1.0, 0.0], 1)
    assert np.isfinite(loss)
    assert loss > 20.0


def test_predict_is_argmax():
    assert predict([0.1, 0.7, 0.2]) == 1
    assert predict([0.9, 0.05, 0.05]) == 0
    assert isinstance(predict([0.1, 0.7, 0.2]), int)


def test_accuracy():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)
    assert accuracy([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert isinstance(accuracy([0, 1], [1, 1]), float)


def test_pipeline_lower_loss_when_logit_favors_truth():
    # raise the logit of the true class -> softmax gives it more prob -> cross-entropy drops
    true = 2
    before = cross_entropy(softmax([1.0, 1.0, 1.0]), true)
    after = cross_entropy(softmax([1.0, 1.0, 3.0]), true)
    assert after < before
    assert predict(softmax([1.0, 1.0, 3.0])) == true
