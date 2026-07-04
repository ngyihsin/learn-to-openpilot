"""Auto-grader for Day 20 — profiling & performance.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

Correctness is checked against the (slow but obviously-correct) naive versions on small inputs;
performance is checked on large inputs where an O(n^2) solution would blow the time budget.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time


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
has_duplicate, common_elements = _impl.has_duplicate, _impl.common_elements


# Obviously-correct, slow reference implementations.
def naive_has_duplicate(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return True
    return False


def naive_common(a, b):
    out = set()
    for x in a:
        for y in b:
            if x == y:
                out.add(x)
    return out


def test_has_duplicate_correct():
    assert has_duplicate([1, 2, 3, 2]) is True
    assert has_duplicate([1, 2, 3, 4]) is False
    assert has_duplicate([]) is False
    for data in ([5, 5], list(range(50)), [1, 2, 3, 3, 4], ["a", "b", "a"]):
        assert has_duplicate(data) == naive_has_duplicate(data)


def test_common_elements_correct():
    assert common_elements([1, 2, 3], [2, 3, 4]) == {2, 3}
    assert common_elements([1, 2], [3, 4]) == set()
    a, b = list(range(0, 60)), list(range(40, 100))
    assert common_elements(a, b) == naive_common(a, b)


def test_has_duplicate_is_linear():
    # 50k distinct items -> worst case (must scan all). O(n^2) would be ~2.5e9 ops; O(n) is instant.
    data = list(range(50_000))
    start = time.perf_counter()
    assert has_duplicate(data) is False
    assert time.perf_counter() - start < 1.0, "too slow — still O(n^2)?"


def test_common_elements_is_linear():
    a = list(range(50_000))
    b = list(range(25_000, 75_000))
    start = time.perf_counter()
    result = common_elements(a, b)
    assert time.perf_counter() - start < 1.0, "too slow — still O(n*m)?"
    assert result == set(range(25_000, 50_000))
