"""Day 03 reference solution — Stack, Queue (two stacks), and RingBuffer."""
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
        self._data.append(value)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError("pop from empty Stack")
        return self._data.pop()

    def peek(self) -> Any:
        if not self._data:
            raise IndexError("peek into empty Stack")
        return self._data[-1]


class Queue:
    def __init__(self) -> None:
        self._in: list[Any] = []
        self._out: list[Any] = []

    def __len__(self) -> int:
        return len(self._in) + len(self._out)

    def is_empty(self) -> bool:
        return len(self) == 0

    def enqueue(self, value: Any) -> None:
        self._in.append(value)

    def _shift(self) -> None:
        if not self._out:
            while self._in:
                self._out.append(self._in.pop())

    def dequeue(self) -> Any:
        self._shift()
        if not self._out:
            raise IndexError("dequeue from empty Queue")
        return self._out.pop()

    def peek(self) -> Any:
        self._shift()
        if not self._out:
            raise IndexError("peek into empty Queue")
        return self._out[-1]


class RingBuffer:
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
        if self.is_full():
            raise IndexError("push to full RingBuffer")
        tail = (self._head + self._count) % self._cap
        self._data[tail] = value
        self._count += 1

    def pop(self) -> Any:
        if self.is_empty():
            raise IndexError("pop from empty RingBuffer")
        value = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % self._cap
        self._count -= 1
        return value
