"""Auto-grader for Day 15 — bump allocator.

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


BumpAllocator = _load(os.environ.get("LP_IMPL", "homework")).BumpAllocator


def test_sequential_allocations():
    a = BumpAllocator(100)
    assert a.alloc(10) == 0
    assert a.alloc(20) == 10
    assert a.alloc(5) == 30
    assert a.used == 35
    assert a.remaining == 65


def test_alignment_rounds_up():
    a = BumpAllocator(100)
    assert a.alloc(1) == 0           # cursor now at 1
    off = a.alloc(8, align=8)        # must round 1 up to 8
    assert off == 8
    assert a.used == 16
    off2 = a.alloc(4, align=4)       # 16 is already aligned to 4
    assert off2 == 16


def test_out_of_memory_raises():
    a = BumpAllocator(16)
    a.alloc(10)
    with pytest.raises(MemoryError):
        a.alloc(10)                  # only 6 left
    assert a.used == 10              # failed alloc doesn't advance the cursor


def test_reset_rewinds_but_keeps_high_water():
    a = BumpAllocator(100)
    a.alloc(40)
    a.alloc(30)
    assert a.high_water == 70
    a.reset()
    assert a.used == 0
    assert a.high_water == 70        # peak usage remembered across resets
    assert a.alloc(10) == 0          # arena reused from the start


@pytest.mark.parametrize("bad", [0, -4])
def test_bad_capacity(bad):
    with pytest.raises(ValueError):
        BumpAllocator(bad)


def test_bad_align():
    a = BumpAllocator(100)
    with pytest.raises(ValueError):
        a.alloc(4, align=0)


def test_zero_size_alloc_ok():
    a = BumpAllocator(8)
    assert a.alloc(0) == 0
    assert a.used == 0
