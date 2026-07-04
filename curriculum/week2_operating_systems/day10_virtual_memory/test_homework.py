"""Auto-grader for Day 10 — Virtual memory & page replacement.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

Fault counts below are hand-computed for the classic Belady reference string.
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
translate, fifo, lru = _impl.translate, _impl.fifo, _impl.lru
# Take the exception class from the implementation under test so identity matches what it
# raises (solution.py imports PageFault from the shared `homework` module).
PageFault = _impl.PageFault

# Belady's classic reference string.
BELADY = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]


# ---- address translation ----

def test_translate_basic():
    # page_size 256, page 1 -> frame 2, offset 100  =>  1*256+100=356  ->  2*256+100=612
    assert translate(356, 256, {0: 5, 1: 2}) == 612


def test_translate_offset_and_frame0():
    assert translate(0, 256, {0: 0}) == 0
    assert translate(255, 256, {0: 7}) == 7 * 256 + 255


def test_translate_unmapped_page_faults():
    with pytest.raises(PageFault):
        translate(9 * 256 + 3, 256, {0: 1})


# ---- FIFO ----

def test_fifo_three_frames():
    r = fifo(BELADY, 3)
    assert r.faults == 9
    assert r.hits == len(BELADY) - 9


def test_fifo_beladys_anomaly():
    # The famous counterintuitive result: giving FIFO *more* frames makes it *worse* here.
    assert fifo(BELADY, 3).faults == 9
    assert fifo(BELADY, 4).faults == 10
    assert fifo(BELADY, 4).faults > fifo(BELADY, 3).faults


# ---- LRU ----

def test_lru_three_frames():
    assert lru(BELADY, 3).faults == 10


def test_lru_never_worse_with_more_frames():
    # LRU is a "stack algorithm" — it can't suffer Belady's anomaly.
    for n in range(1, 6):
        assert lru(BELADY, n).faults >= lru(BELADY, n + 1).faults


# ---- invariants that must hold for any policy ----

@pytest.mark.parametrize("policy", ["fifo", "lru"])
def test_bookkeeping_consistent(policy):
    run = {"fifo": fifo, "lru": lru}[policy]
    r = run(BELADY, 3)
    assert r.faults + r.hits == len(BELADY)
    assert len(r.fault_flags) == len(BELADY)
    assert sum(r.fault_flags) == r.faults
    # Never hold more pages than we have frames.
    assert all(len(snapshot) <= 3 for snapshot in r.frames_over_time)


def test_all_faults_when_frames_scarce():
    # One frame + all-distinct pages -> every reference faults.
    refs = [1, 2, 3, 4, 5]
    assert fifo(refs, 1).faults == 5
    assert lru(refs, 1).faults == 5


def test_no_faults_after_warmup_single_page():
    r = lru([7, 7, 7, 7], 2)
    assert r.faults == 1        # only the first touch faults
    assert r.hits == 3
