"""Day 20 homework — profiling & fixing a 100x slowdown.

The first rule of performance: *measure, don't guess.* You profile to find the hot spot, then
you fix the algorithm — usually by removing a nested loop. Today you're handed two functions that
are correct but quadratic (O(n²)) and you rewrite them to be linear (O(n)) using the right data
structure (a set/hash). The grader checks you still get the right answer **and** that you're
actually fast on large inputs — a leftover O(n²) solution will time out.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from typing import Any


def has_duplicate(items: list[Any]) -> bool:
    """Return True if any value appears more than once.

    The naive version compares every pair — O(n²). Do it in O(n) by remembering what you've
    seen in a set as you go.
    """
    # TODO: keep a `seen` set; return True the moment you re-encounter a value; else False
    raise NotImplementedError


def common_elements(a: list[Any], b: list[Any]) -> set:
    """Return the set of values present in BOTH `a` and `b`.

    The naive version is a nested loop — O(n·m). Sets give you O(n+m).
    """
    # TODO: convert to sets and intersect
    raise NotImplementedError
