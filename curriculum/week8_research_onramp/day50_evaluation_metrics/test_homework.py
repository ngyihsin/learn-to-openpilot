"""Auto-grader for Day 50 — evaluation metrics done right.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys

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
confusion = _impl.confusion
precision = _impl.precision
recall = _impl.recall
f1_score = _impl.f1_score


def test_confusion_counts():
    assert confusion([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]) == (2, 1, 1, 1)


def test_precision_recall_f1():
    tp, fp, fn, tn = confusion([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert precision(tp, fp) == pytest.approx(2 / 3)
    assert recall(tp, fn) == pytest.approx(2 / 3)
    assert f1_score(precision(tp, fp), recall(tp, fn)) == pytest.approx(2 / 3)


def test_perfect_and_zero():
    tp, fp, fn, tn = confusion([1, 0, 1, 0], [1, 0, 1, 0])
    assert precision(tp, fp) == 1.0 and recall(tp, fn) == 1.0
    assert f1_score(1.0, 1.0) == 1.0


def test_edge_cases_no_divide_by_zero():
    assert precision(0, 0) == 0.0       # nothing predicted positive
    assert recall(0, 0) == 0.0          # no real positives
    assert f1_score(0.0, 0.0) == 0.0    # both zero


def test_accuracy_trap_is_caught_by_recall():
    # 5 positives in 100, model predicts ALL negative: ~95% "accurate" but useless.
    y_true = [1] * 5 + [0] * 95
    y_pred = [0] * 100
    tp, fp, fn, tn = confusion(y_true, y_pred)
    assert recall(tp, fn) == 0.0        # recall exposes the failure accuracy hides
    assert (tp + tn) / 100 == 0.95      # ...even though "accuracy" looks great
