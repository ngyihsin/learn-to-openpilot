"""Auto-grader for Day 08 — CPU scheduling.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

We check the exact Gantt timeline *and* the summary metrics against hand-computed values
for a classic three-process workload.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)  # let solution.py do `from homework import ...`
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load(os.environ.get("LP_IMPL", "homework"))
_types = _load("homework")  # Process/Segment always come from the shared homework module
Process, Segment = _types.Process, _types.Segment
fcfs, sjf, round_robin = _impl.fcfs, _impl.sjf, _impl.round_robin


# Classic workload:  P1(arr=0,burst=5)  P2(arr=1,burst=3)  P3(arr=2,burst=1)
WORKLOAD = [Process("P1", 0, 5), Process("P2", 1, 3), Process("P3", 2, 1)]


def timeline(res):
    return [(s.pid, s.start, s.end) for s in res.segments]


def test_fcfs_timeline_and_metrics():
    r = fcfs(WORKLOAD)
    assert timeline(r) == [("P1", 0, 5), ("P2", 5, 8), ("P3", 8, 9)]
    assert r.waiting == {"P1": 0, "P2": 4, "P3": 6}
    assert r.turnaround == {"P1": 5, "P2": 7, "P3": 7}
    assert r.avg_waiting == pytest.approx(10 / 3)


def test_sjf_prefers_short_jobs():
    r = sjf(WORKLOAD)
    # After P1 finishes at t=5, both P2 and P3 have arrived; the 1-tick P3 goes first.
    assert timeline(r) == [("P1", 0, 5), ("P3", 5, 6), ("P2", 6, 9)]
    assert r.avg_waiting == pytest.approx(8 / 3)
    # SJF should beat FCFS on average waiting time for this workload.
    assert r.avg_waiting < fcfs(WORKLOAD).avg_waiting


def test_round_robin_quantum_2():
    r = round_robin(WORKLOAD, quantum=2)
    assert timeline(r) == [
        ("P1", 0, 2), ("P2", 2, 4), ("P3", 4, 5),
        ("P1", 5, 7), ("P2", 7, 8), ("P1", 8, 9),
    ]
    assert r.completion == {"P1": 9, "P2": 8, "P3": 5}
    assert r.waiting == {"P1": 4, "P2": 4, "P3": 2}


def test_round_robin_large_quantum_is_fcfs():
    # A quantum bigger than every burst means nobody is ever preempted -> same as FCFS.
    big = round_robin(WORKLOAD, quantum=100)
    assert timeline(big) == timeline(fcfs(WORKLOAD))


def test_round_robin_rejects_bad_quantum():
    with pytest.raises(ValueError):
        round_robin(WORKLOAD, quantum=0)


def test_all_bursts_fully_served():
    # Whatever the policy, every process must get exactly `burst` ticks of CPU.
    for sched in (fcfs, sjf, lambda w: round_robin(w, 2)):
        r = sched(WORKLOAD)
        served: dict[str, int] = {}
        for s in r.segments:
            served[s.pid] = served.get(s.pid, 0) + (s.end - s.start)
        assert served == {"P1": 5, "P2": 3, "P3": 1}
