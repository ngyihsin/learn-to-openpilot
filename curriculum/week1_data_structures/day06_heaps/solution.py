"""Day 06 reference solution — binary min-heap."""
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

    def _sift_up(self, i: int) -> None:
        data = self._data
        while i > 0:
            parent = (i - 1) // 2
            if data[i] < data[parent]:
                data[i], data[parent] = data[parent], data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        data = self._data
        n = len(data)
        while True:
            smallest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < n and data[left] < data[smallest]:
                smallest = left
            if right < n and data[right] < data[smallest]:
                smallest = right
            if smallest == i:
                break
            data[i], data[smallest] = data[smallest], data[i]
            i = smallest

    def push(self, value: Any) -> None:
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError("pop from empty heap")
        data = self._data
        minimum = data[0]
        last = data.pop()
        if data:
            data[0] = last
            self._sift_down(0)
        return minimum

    @classmethod
    def heapify(cls, values: Iterable[Any]) -> "MinHeap":
        heap = cls()
        heap._data = list(values)
        for i in range(len(heap._data) // 2 - 1, -1, -1):
            heap._sift_down(i)
        return heap
