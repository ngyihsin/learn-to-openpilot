"""Day 12 reference solution — SafeCounter and BoundedBuffer."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any


class SafeCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class BoundedBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = capacity
        self._items: deque = deque()
        self._cv = threading.Condition()
        self.max_observed = 0

    def __len__(self) -> int:
        with self._cv:
            return len(self._items)

    def put(self, item: Any) -> None:
        with self._cv:
            while len(self._items) >= self._cap:
                self._cv.wait()
            self._items.append(item)
            if len(self._items) > self.max_observed:
                self.max_observed = len(self._items)
            self._cv.notify_all()

    def get(self) -> Any:
        with self._cv:
            while not self._items:
                self._cv.wait()
            item = self._items.popleft()
            self._cv.notify_all()
            return item
