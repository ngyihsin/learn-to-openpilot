"""Auto-grader for Day 04 — HashMap (open addressing).

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


HashMap = _load(os.environ.get("LP_IMPL", "homework")).HashMap


def test_put_get_and_update():
    m = HashMap()
    m.put("a", 1)
    m.put("b", 2)
    assert m.get("a") == 1 and m.get("b") == 2
    assert len(m) == 2
    m.put("a", 99)                 # update, not insert
    assert m.get("a") == 99
    assert len(m) == 2


def test_missing_key_raises():
    m = HashMap()
    with pytest.raises(KeyError):
        m.get("nope")
    assert "nope" not in m
    m.put("x", 1)
    assert "x" in m


def test_linear_probing_collisions():
    # int keys hash to themselves; with capacity 8, 1/9/17 all map to index 1.
    m = HashMap(initial_capacity=8)
    for k in (1, 9, 17):
        m.put(k, k * 10)
    assert m.capacity == 8         # too few entries to trigger a resize
    assert [m.get(1), m.get(9), m.get(17)] == [10, 90, 170]


def test_delete_preserves_probe_chain():
    # Remove the MIDDLE of a collision chain; the others must still be findable (tombstone).
    m = HashMap(initial_capacity=8)
    for k in (1, 9, 17):
        m.put(k, k)
    m.remove(9)
    assert 9 not in m
    assert m.get(1) == 1 and m.get(17) == 17    # chain not broken
    assert len(m) == 2
    with pytest.raises(KeyError):
        m.remove(9)                              # removing again raises


def test_tombstone_slot_is_reused():
    m = HashMap(initial_capacity=8)
    m.put(1, 1)
    m.put(9, 9)
    m.remove(1)                    # slot 1 becomes a tombstone
    m.put(1, 111)                  # should reuse it, not grow len oddly
    assert m.get(1) == 111
    assert m.get(9) == 9
    assert len(m) == 2


def test_resize_grows_and_preserves_everything():
    m = HashMap(initial_capacity=8)
    for i in range(100):
        m.put(i, i * i)
    assert len(m) == 100
    assert m.capacity > 8          # must have resized
    assert m.load_factor <= 0.7    # resizing keeps it below the threshold
    assert all(m.get(i) == i * i for i in range(100))


def test_items_roundtrip():
    m = HashMap()
    data = {"one": 1, "two": 2, "three": 3}
    for k, v in data.items():
        m.put(k, v)
    assert dict(m.items()) == data
