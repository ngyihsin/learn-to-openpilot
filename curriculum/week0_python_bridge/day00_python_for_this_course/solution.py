"""Day 00 reference solution — the Python this course is written in.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.
"""
from __future__ import annotations


class Backpack:
    """A container with a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items = []

    def add(self, item) -> None:
        if len(self.items) >= self.capacity:
            raise ValueError("backpack is full")
        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        if not 0 <= i < len(self.items):
            raise IndexError(f"index {i} out of range")
        return self.items[i]

    @property
    def is_full(self) -> bool:
        return len(self.items) >= self.capacity


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("cannot divide by zero")
    return a / b
