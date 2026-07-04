"""Auto-grader for Day 13 — deadlock detection.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys


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
find_deadlock_cycle = _impl.find_deadlock_cycle
has_deadlock = _impl.has_deadlock
safe_lock_order = _impl.safe_lock_order


def _is_valid_cycle(cycle, wait_for) -> bool:
    """Each node in the returned cycle must actually wait for the next (wrapping around)."""
    if not cycle:
        return False
    n = len(cycle)
    return all(cycle[(i + 1) % n] in wait_for.get(cycle[i], []) for i in range(n))


def test_no_deadlock_acyclic():
    wait_for = {"A": ["B"], "B": ["C"], "C": []}
    assert has_deadlock(wait_for) is False
    assert find_deadlock_cycle(wait_for) is None


def test_two_process_deadlock():
    wait_for = {"A": ["B"], "B": ["A"]}      # classic AB-BA deadlock
    assert has_deadlock(wait_for) is True
    cycle = find_deadlock_cycle(wait_for)
    assert set(cycle) == {"A", "B"}
    assert _is_valid_cycle(cycle, wait_for)


def test_longer_cycle():
    wait_for = {"A": ["B"], "B": ["C"], "C": ["A"], "D": ["A"]}   # D isn't part of the cycle
    assert has_deadlock(wait_for) is True
    cycle = find_deadlock_cycle(wait_for)
    assert set(cycle) == {"A", "B", "C"}
    assert _is_valid_cycle(cycle, wait_for)


def test_self_wait_is_deadlock():
    wait_for = {"A": ["A"]}
    assert has_deadlock(wait_for) is True
    assert _is_valid_cycle(find_deadlock_cycle(wait_for), wait_for)


def test_disjoint_graph_one_cycle():
    wait_for = {"A": ["B"], "B": [], "X": ["Y"], "Y": ["X"]}
    assert has_deadlock(wait_for) is True
    assert set(find_deadlock_cycle(wait_for)) == {"X", "Y"}


def test_safe_lock_order_is_consistent():
    locks = ["mutex_c", "mutex_a", "mutex_b"]
    order = safe_lock_order(locks)
    assert order == sorted(locks)                 # deterministic global order
    assert sorted(order) == sorted(locks)         # a permutation, nothing dropped
    # Two threads sorting their (possibly different-order) lock sets get the SAME order.
    assert safe_lock_order(["mutex_b", "mutex_a"]) == safe_lock_order(["mutex_a", "mutex_b"])
