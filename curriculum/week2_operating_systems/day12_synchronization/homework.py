"""Day 12 homework — Locks, Conditions & the producer/consumer problem.

When threads *share* mutable state (Day 11 avoided this on purpose), you need synchronization.
`x += 1` is not atomic — it's load, add, store — so two threads can interleave and lose an
update. A **lock** makes a section mutually exclusive. A **condition variable** lets a thread
*wait* efficiently until some predicate holds (buffer not full / not empty) instead of spinning.

You'll build a thread-safe counter and the classic **bounded buffer** (producer/consumer), the
pattern behind every work queue and pipeline — including the message queues in a robotics stack.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any


class SafeCounter:
    """A counter that stays correct under many concurrent incrementers."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, n: int = 1) -> None:
        # TODO: acquire self._lock (use `with`) around the read-modify-write of self._value
        raise NotImplementedError

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class BoundedBuffer:
    """A fixed-capacity thread-safe FIFO. `put` blocks while full; `get` blocks while empty."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = capacity
        self._items: deque = deque()
        self._cv = threading.Condition()
        self.max_observed = 0     # high-water mark; the grader checks it never exceeds capacity

    def __len__(self) -> int:
        with self._cv:
            return len(self._items)

    def put(self, item: Any) -> None:
        # TODO:
        #   with self._cv:
        #     while the buffer is full: self._cv.wait()
        #     append the item; update self.max_observed; self._cv.notify_all()
        raise NotImplementedError

    def get(self) -> Any:
        # TODO:
        #   with self._cv:
        #     while the buffer is empty: self._cv.wait()
        #     pop the oldest item; self._cv.notify_all(); return it
        raise NotImplementedError
