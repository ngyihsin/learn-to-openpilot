"""Auto-grader for Day 21 — build systems.

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
build_order, needs_rebuild, rebuild_set = _impl.build_order, _impl.needs_rebuild, _impl.rebuild_set

# app depends on two objects, each compiled from a source.
DEPS = {
    "app": ["main.o", "util.o"],
    "main.o": ["main.c", "shared.h"],
    "util.o": ["util.c", "shared.h"],
    "main.c": [],
    "util.c": [],
    "shared.h": [],
}


def test_build_order_respects_dependencies():
    order = build_order(DEPS)
    pos = {name: i for i, name in enumerate(order)}
    assert set(order) == set(DEPS)                       # every node present
    for target, prereqs in DEPS.items():
        for p in prereqs:
            assert pos[p] < pos[target]                  # prereq built before target


def test_build_order_detects_cycle():
    with pytest.raises(ValueError):
        build_order({"a": ["b"], "b": ["c"], "c": ["a"]})


def test_needs_rebuild_missing_target():
    assert needs_rebuild("app", DEPS, {"main.o": 1, "util.o": 1}) is True   # app never built


def test_needs_rebuild_stale_prereq():
    # main.o built at t=5, but main.c edited at t=9 -> stale.
    mtimes = {"main.o": 5, "main.c": 9, "shared.h": 1}
    assert needs_rebuild("main.o", DEPS, mtimes) is True


def test_needs_rebuild_up_to_date():
    mtimes = {"main.o": 10, "main.c": 5, "shared.h": 3}
    assert needs_rebuild("main.o", DEPS, mtimes) is False


def test_rebuild_set_from_source_change():
    # Touching main.c forces main.o and then app to rebuild (not util.o).
    assert rebuild_set(DEPS, {"main.c"}) == {"main.o", "app"}


def test_rebuild_set_shared_header_hits_everything():
    # shared.h feeds both objects -> everything downstream rebuilds.
    assert rebuild_set(DEPS, {"shared.h"}) == {"main.o", "util.o", "app"}


def test_rebuild_set_no_changes():
    assert rebuild_set(DEPS, set()) == set()
