"""Auto-grader for Day 03 — Stack, Queue, RingBuffer.

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
Stack, Queue, RingBuffer = _impl.Stack, _impl.Queue, _impl.RingBuffer


# ---- Stack (LIFO) ----

def test_stack_lifo():
    s = Stack()
    for v in (1, 2, 3):
        s.push(v)
    assert len(s) == 3
    assert s.peek() == 3
    assert [s.pop(), s.pop(), s.pop()] == [3, 2, 1]
    assert s.is_empty()


def test_stack_empty_raises():
    s = Stack()
    with pytest.raises(IndexError):
        s.pop()
    with pytest.raises(IndexError):
        s.peek()


# ---- Queue (FIFO) ----

def test_queue_fifo():
    q = Queue()
    for v in (1, 2, 3):
        q.enqueue(v)
    assert q.peek() == 1
    assert [q.dequeue(), q.dequeue(), q.dequeue()] == [1, 2, 3]
    assert q.is_empty()


def test_queue_interleaved():
    q = Queue()
    q.enqueue("a"); q.enqueue("b")
    assert q.dequeue() == "a"
    q.enqueue("c")               # forces a mix across the two internal stacks
    assert [q.dequeue(), q.dequeue()] == ["b", "c"]
    assert len(q) == 0


def test_queue_empty_raises():
    with pytest.raises(IndexError):
        Queue().dequeue()


# ---- RingBuffer ----

def test_ring_basic_fifo_and_full():
    rb = RingBuffer(3)
    assert rb.is_empty() and rb.capacity == 3
    rb.push(1); rb.push(2); rb.push(3)
    assert rb.is_full() and len(rb) == 3
    with pytest.raises(IndexError):
        rb.push(4)                # full
    assert [rb.pop(), rb.pop(), rb.pop()] == [1, 2, 3]
    assert rb.is_empty()
    with pytest.raises(IndexError):
        rb.pop()                  # empty


def test_ring_wraparound_reuses_slots():
    rb = RingBuffer(3)
    rb.push(1); rb.push(2); rb.push(3)
    assert rb.pop() == 1          # head advances
    assert rb.pop() == 2
    rb.push(4); rb.push(5)        # these wrap into the freed slots
    assert len(rb) == 3
    assert [rb.pop(), rb.pop(), rb.pop()] == [3, 4, 5]


def test_ring_capacity_is_fixed():
    rb = RingBuffer(2)
    for _ in range(50):           # lots of churn, memory stays capacity-bounded
        rb.push(1)
        rb.push(2)
        assert len(rb) <= rb.capacity
        rb.pop(); rb.pop()


def test_ring_bad_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)
