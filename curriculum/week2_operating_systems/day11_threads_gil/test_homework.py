"""Auto-grader for Day 11 — threads & work-splitting.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

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
chunk, threaded_sum, parallel_map = _impl.chunk, _impl.threaded_sum, _impl.parallel_map


def test_chunk_sizes_and_reconstruction():
    items = list(range(10))
    parts = chunk(items, 3)
    assert len(parts) == 3
    assert [len(p) for p in parts] == [4, 3, 3]   # sizes differ by at most 1
    flat = [x for p in parts for x in p]
    assert flat == items                          # concatenation reproduces the input


def test_chunk_more_buckets_than_items():
    parts = chunk([1, 2], 4)
    assert len(parts) == 4
    assert [x for p in parts for x in p] == [1, 2]
    assert [len(p) for p in parts] == [1, 1, 0, 0]


def test_chunk_bad_k():
    with pytest.raises(ValueError):
        chunk([1, 2, 3], 0)


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_threaded_sum_matches_builtin(workers):
    nums = list(range(1000))
    assert threaded_sum(nums, workers) == sum(nums)


def test_threaded_sum_empty():
    assert threaded_sum([], 4) == 0


def test_parallel_map_preserves_order():
    items = list(range(20))
    assert parallel_map(lambda x: x * x, items, workers=4) == [x * x for x in items]


def test_parallel_map_order_despite_uneven_timing():
    # Earlier items sleep longer, so they finish LAST — yet the output must stay in input order.
    def slow(x):
        time.sleep((5 - x) * 0.01)
        return x
    assert parallel_map(slow, [0, 1, 2, 3, 4], workers=5) == [0, 1, 2, 3, 4]


def test_parallel_map_empty():
    assert parallel_map(lambda x: x, [], workers=4) == []
