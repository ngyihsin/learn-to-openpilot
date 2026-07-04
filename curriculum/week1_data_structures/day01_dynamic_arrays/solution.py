"""Day 01 reference solution — a doubling DynamicArray.

Try the homework yourself before reading this. If you're comparing your work: the key
ideas are (1) grow by *doubling* so appends are amortized O(1), and (2) never index past
the logical size ``self._n`` even though ``self._cap`` slots are allocated.
"""
from __future__ import annotations

from typing import Any


class DynamicArray:
    def __init__(self) -> None:
        self._cap = 1
        self._n = 0
        self._data: list[Any] = [None] * self._cap
        self.resizes = 0

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return self._cap

    def _resize(self, new_cap: int) -> None:
        new_data = [None] * new_cap
        for i in range(self._n):
            new_data[i] = self._data[i]
        self._data = new_data
        self._cap = new_cap
        self.resizes += 1

    def append(self, value: Any) -> None:
        if self._n == self._cap:
            self._resize(2 * self._cap)
        self._data[self._n] = value
        self._n += 1

    def _check(self, i: int) -> None:
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for length {self._n}")

    def __getitem__(self, i: int) -> Any:
        self._check(i)
        return self._data[i]

    def __setitem__(self, i: int, value: Any) -> None:
        self._check(i)
        self._data[i] = value

    def pop(self) -> Any:
        if self._n == 0:
            raise IndexError("pop from empty DynamicArray")
        self._n -= 1
        value = self._data[self._n]
        self._data[self._n] = None
        # Shrink when the array is only a quarter full — keeps memory O(n).
        if self._n > 0 and self._n <= self._cap // 4:
            self._resize(max(1, self._cap // 2))
        return value

    def __repr__(self) -> str:
        return f"DynamicArray([{', '.join(repr(self._data[i]) for i in range(self._n))}])"
