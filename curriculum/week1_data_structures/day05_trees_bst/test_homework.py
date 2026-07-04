"""Auto-grader for Day 05 — Binary Search Tree.

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


BST = _load(os.environ.get("LP_IMPL", "homework")).BST

VALUES = [5, 3, 8, 1, 4, 7, 9, 2, 6]


def make(values=VALUES) -> "BST":
    t = BST()
    for v in values:
        t.insert(v)
    return t


def test_in_order_is_sorted():
    t = make()
    assert t.in_order() == sorted(VALUES)
    assert len(t) == len(VALUES)


def test_duplicates_ignored():
    t = make()
    t.insert(5); t.insert(8)
    assert len(t) == len(VALUES)
    assert t.in_order() == sorted(VALUES)


def test_contains():
    t = make()
    for v in VALUES:
        assert t.contains(v)
    for v in (0, 10, 42):
        assert not t.contains(v)


def test_min_max():
    t = make()
    assert t.min() == 1
    assert t.max() == 9


def test_height_degenerate_vs_balanced():
    degenerate = make([1, 2, 3, 4, 5])     # inserting sorted -> a linked list
    assert degenerate.height() == 4        # n-1
    balanced = make([4, 2, 6, 1, 3, 5, 7])
    assert balanced.height() == 2          # perfectly balanced 7-node tree


@pytest.mark.parametrize("victim", [2, 1, 3, 8, 5, 9])
def test_delete_all_cases_keep_bst_sorted(victim):
    # victims cover leaf, one-child, two-children, and the root.
    t = make()
    assert t.delete(victim) is True
    remaining = sorted(v for v in VALUES if v != victim)
    assert t.in_order() == remaining       # still sorted & correct set
    assert len(t) == len(remaining)
    assert not t.contains(victim)


def test_delete_missing_returns_false():
    t = make()
    assert t.delete(999) is False
    assert len(t) == len(VALUES)


def test_delete_everything():
    t = make()
    for v in VALUES:
        assert t.delete(v) is True
    assert len(t) == 0
    assert t.in_order() == []
