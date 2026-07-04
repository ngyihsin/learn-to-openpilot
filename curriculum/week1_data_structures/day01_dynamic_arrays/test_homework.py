"""Auto-grader for Day 01 — DynamicArray.

Run ``pytest -q`` here. It grades your ``homework.py``.
To grade the reference solution instead:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest


def _load(name: str):
    """Load a sibling module (homework.py / solution.py) under a unique name,
    so the whole 30-day suite can run without module-name collisions."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


DynamicArray = _load(os.environ.get("LP_IMPL", "homework")).DynamicArray


def test_starts_empty():
    a = DynamicArray()
    assert len(a) == 0


def test_append_and_index():
    a = DynamicArray()
    for i in range(5):
        a.append(i * 10)
    assert len(a) == 5
    assert [a[i] for i in range(5)] == [0, 10, 20, 30, 40]


def test_setitem_overwrites():
    a = DynamicArray()
    a.append("x")
    a[0] = "y"
    assert a[0] == "y"
    assert len(a) == 1


@pytest.mark.parametrize("bad", [-1, 0, 1, 100])
def test_index_out_of_range_raises(bad):
    a = DynamicArray()          # empty -> every index is out of range
    with pytest.raises(IndexError):
        _ = a[bad]


def test_pop_returns_last_and_shrinks_length():
    a = DynamicArray()
    for v in (1, 2, 3):
        a.append(v)
    assert a.pop() == 3
    assert a.pop() == 2
    assert len(a) == 1
    assert a[0] == 1


def test_pop_from_empty_raises():
    a = DynamicArray()
    with pytest.raises(IndexError):
        a.pop()


def test_capacity_at_least_length():
    a = DynamicArray()
    for i in range(50):
        a.append(i)
        assert a.capacity >= len(a)


def test_growth_is_doubling_not_linear():
    """Capacity must stay within a constant factor of length — i.e. you doubled,
    you didn't grow by a fixed +k each time."""
    a = DynamicArray()
    for i in range(1000):
        a.append(i)
    assert len(a) == 1000
    # A doubling array holds n items in a block of size < 2n.
    assert a.capacity < 2 * len(a)


def test_amortized_constant_appends():
    """1000 appends should trigger only ~log2(1000) ≈ 10 resizes, not ~1000.
    This is what makes append amortized O(1)."""
    a = DynamicArray()
    for i in range(1000):
        a.append(i)
    assert a.resizes <= 2 * math.log2(1000) + 2   # generous: ~22, fails hard for linear growth
