"""Day 06 homework — a binary min-heap (priority queue).

A heap is a *complete* binary tree flattened into an array: for a node at index `i`, its
children live at `2i+1` and `2i+2`, and every parent is <= its children (a **min**-heap). That
gives you the smallest element at index 0 in O(1), and push/pop in O(log n) by "sifting" an
element up or down to restore the order.

Priority queues are the backbone of Dijkstra, A*, event simulation, and the SJF scheduler you'll
build on Day 08. Implement the sift operations and you've implemented all of it.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from typing import Any, Iterable


class MinHeap:
    def __init__(self) -> None:
        self._data: list[Any] = []

    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return not self._data

    def peek(self) -> Any:
        if not self._data:
            raise IndexError("peek into empty heap")
        return self._data[0]

    # ----- implement below -----

    def _sift_up(self, i: int) -> None:
        """Bubble the element at index i toward the root while it's smaller than its parent."""
        # TODO: parent = (i - 1) // 2; while i > 0 and data[i] < data[parent]: swap, move up
        raise NotImplementedError

    def _sift_down(self, i: int) -> None:
        """Push the element at index i toward the leaves while a child is smaller than it."""
        # TODO: repeatedly find the smaller of children (2i+1, 2i+2 if in range); if it's
        #       smaller than data[i], swap and continue there; else stop.
        raise NotImplementedError

    def push(self, value: Any) -> None:
        # TODO: append to the end, then _sift_up from the last index
        raise NotImplementedError

    def pop(self) -> Any:
        """Remove and return the minimum. Raise IndexError if empty."""
        # TODO: the min is data[0]; move the last element to index 0, shrink, _sift_down(0);
        #       handle the single/empty cases carefully.
        raise NotImplementedError

    @classmethod
    def heapify(cls, values: Iterable[Any]) -> "MinHeap":
        """Build a heap from an iterable in O(n) (faster than n pushes) by sifting down every
        non-leaf node from the bottom up."""
        heap = cls()
        heap._data = list(values)
        # TODO: for i from the last parent (len//2 - 1) down to 0: heap._sift_down(i)
        raise NotImplementedError
