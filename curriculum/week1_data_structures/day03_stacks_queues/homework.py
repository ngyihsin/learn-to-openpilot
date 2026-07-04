"""Day 03 homework — Stacks, Queues & a Ring Buffer.

Three of the most-used containers in systems code, all built on the array/linked ideas from
Days 01–02:

- **Stack** — Last-In-First-Out (LIFO). Function calls, undo, DFS.
- **Queue** — First-In-First-Out (FIFO). Task/message queues, BFS. You'll build it from *two
  stacks* so dequeue is amortized O(1) — a classic trick.
- **RingBuffer** — a fixed-capacity FIFO in a reused block of memory. This is how real-time
  systems (openpilot included) pass a bounded stream of messages without ever allocating.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from typing import Any


class Stack:
    def __init__(self) -> None:
        self._data: list[Any] = []

    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return not self._data

    def push(self, value: Any) -> None:
        # TODO: add to the top
        raise NotImplementedError

    def pop(self) -> Any:
        # TODO: remove & return the top; raise IndexError if empty
        raise NotImplementedError

    def peek(self) -> Any:
        # TODO: return the top without removing; raise IndexError if empty
        raise NotImplementedError


class Queue:
    """FIFO queue backed by two stacks: push onto `_in`, pop from `_out`. When `_out` is
    empty, tip everything from `_in` into `_out` (which reverses it) — each item moves at most
    once, so dequeue is amortized O(1)."""

    def __init__(self) -> None:
        self._in: list[Any] = []
        self._out: list[Any] = []

    def __len__(self) -> int:
        return len(self._in) + len(self._out)

    def is_empty(self) -> bool:
        return len(self) == 0

    def enqueue(self, value: Any) -> None:
        # TODO: push onto _in
        raise NotImplementedError

    def dequeue(self) -> Any:
        # TODO: if _out is empty, move all of _in into _out (reversed), then pop _out;
        #       raise IndexError if the whole queue is empty
        raise NotImplementedError

    def peek(self) -> Any:
        # TODO: like dequeue but don't remove
        raise NotImplementedError


class RingBuffer:
    """A fixed-capacity FIFO living in one reused array. Track a head index and a count;
    the tail is computed as (head + count) % capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = capacity
        self._data: list[Any] = [None] * capacity
        self._head = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._cap

    def is_empty(self) -> bool:
        return self._count == 0

    def is_full(self) -> bool:
        return self._count == self._cap

    def push(self, value: Any) -> None:
        # TODO: raise IndexError if full; write at index (head + count) % cap; bump count
        raise NotImplementedError

    def pop(self) -> Any:
        # TODO: raise IndexError if empty; read at head; advance head = (head + 1) % cap;
        #       drop count; return the value
        raise NotImplementedError
