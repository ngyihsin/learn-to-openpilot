"""Auto-grader for Day 06 — MinHeap.

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


MinHeap = _load(os.environ.get("LP_IMPL", "homework")).MinHeap

DATA = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]


def drain(h) -> list:
    return [h.pop() for _ in range(len(h))]


def test_push_pop_yields_sorted():
    h = MinHeap()
    for v in DATA:
        h.push(v)
    assert len(h) == len(DATA)
    assert h.peek() == min(DATA)
    assert drain(h) == sorted(DATA)     # popping a min-heap gives ascending order
    assert h.is_empty()


def test_heapify_then_drain_is_heapsort():
    h = MinHeap.heapify(DATA)
    assert len(h) == len(DATA)
    assert h.peek() == min(DATA)
    assert drain(h) == sorted(DATA)


def test_interleaved_push_pop_keeps_min_on_top():
    h = MinHeap()
    for v in [5, 2, 8]:
        h.push(v)
    assert h.pop() == 2
    h.push(1)
    h.push(9)
    assert h.peek() == 1
    assert h.pop() == 1
    assert h.pop() == 5
    assert drain(h) == [8, 9]


def test_duplicates_ok():
    h = MinHeap()
    for v in [3, 1, 3, 1, 2, 2]:
        h.push(v)
    assert drain(h) == [1, 1, 2, 2, 3, 3]


def test_empty_raises():
    h = MinHeap()
    with pytest.raises(IndexError):
        h.pop()
    with pytest.raises(IndexError):
        h.peek()


def test_single_element():
    h = MinHeap()
    h.push(42)
    assert h.peek() == 42
    assert h.pop() == 42
    assert h.is_empty()


def test_heap_property_holds_after_ops():
    # Directly assert the array invariant: every parent <= its children.
    h = MinHeap.heapify(DATA)
    h.push(-1)
    h.pop()
    arr = h._data
    for i in range(len(arr)):
        for child in (2 * i + 1, 2 * i + 2):
            if child < len(arr):
                assert arr[i] <= arr[child]
