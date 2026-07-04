"""Auto-grader for Day 12 — synchronization.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

These tests use real threads but stay deterministic: they join everything and check exact
totals and multisets, so a correct (properly-locked) implementation always passes.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading

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
SafeCounter, BoundedBuffer = _impl.SafeCounter, _impl.BoundedBuffer


def test_counter_no_lost_updates():
    counter = SafeCounter()
    threads_n, per_thread = 8, 5000

    def work():
        for _ in range(per_thread):
            counter.increment()

    threads = [threading.Thread(target=work) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter.value == threads_n * per_thread   # no update was lost to a race


def test_counter_increment_by_n():
    c = SafeCounter()
    c.increment(5)
    c.increment(3)
    assert c.value == 8


def test_bad_capacity():
    with pytest.raises(ValueError):
        BoundedBuffer(0)


def test_producer_consumer_transfers_everything_exactly_once():
    capacity = 8
    buf = BoundedBuffer(capacity)
    n_producers, n_consumers, per_producer = 4, 4, 50
    total = n_producers * per_producer
    per_consumer = total // n_consumers

    def produce(pid: int):
        for i in range(per_producer):
            buf.put(pid * 1000 + i)          # globally-unique values

    consumed_lists: list[list] = [[] for _ in range(n_consumers)]

    def consume(cid: int):
        for _ in range(per_consumer):
            consumed_lists[cid].append(buf.get())

    threads = ([threading.Thread(target=produce, args=(p,)) for p in range(n_producers)]
               + [threading.Thread(target=consume, args=(c,)) for c in range(n_consumers)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    produced = {p * 1000 + i for p in range(n_producers) for i in range(per_producer)}
    consumed = [x for lst in consumed_lists for x in lst]
    assert len(consumed) == total
    assert set(consumed) == produced          # nothing lost or duplicated
    assert buf.max_observed <= capacity       # `put` really blocked when full
    assert len(buf) == 0                       # buffer drained


def test_buffer_len_tracks_contents():
    buf = BoundedBuffer(4)
    buf.put("a"); buf.put("b")
    assert len(buf) == 2
    assert buf.get() == "a"
    assert len(buf) == 1
