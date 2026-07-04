"""Day 01 homework — build a DynamicArray.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Stuck? Peek at ``solution.py`` (but try for 20 minutes first!), or ask Claude Code
for a *hint* rather than the answer.

The whole point of today: an array lives in one contiguous block of memory of a fixed
`capacity`. When it fills up, you can't "extend" it — you allocate a bigger block and
copy everything over. Do that by *doubling* the capacity and the average (amortized)
cost of `append` stays O(1), even though individual appends occasionally cost O(n).
"""
from __future__ import annotations

from typing import Any


class DynamicArray:
    def __init__(self) -> None:
        self._cap: int = 1
        self._n: int = 0
        self._data: list[Any] = [None] * self._cap
        # Instrumentation for the amortized-cost lesson — leave this in.
        self.resizes: int = 0

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        """Number of allocated slots (>= len)."""
        return self._cap

    def _resize(self, new_cap: int) -> None:
        """Allocate a new block of size ``new_cap`` and copy existing items over."""
        # TODO:
        #   1. make a new list of size new_cap filled with None
        #   2. copy self._n existing items into it
        #   3. point self._data at it, update self._cap
        #   4. self.resizes += 1   (so the grader can see how often you grow)
        raise NotImplementedError

    def append(self, value: Any) -> None:
        """Add ``value`` to the end. Grow (double) the capacity if full."""
        # TODO: if the array is full (self._n == self._cap), call self._resize(2 * self._cap)
        #       then store value at index self._n and increment self._n
        raise NotImplementedError

    def __getitem__(self, i: int) -> Any:
        # TODO: bounds-check 0 <= i < self._n (raise IndexError otherwise), then return the item
        raise NotImplementedError

    def __setitem__(self, i: int, value: Any) -> None:
        # TODO: bounds-check, then overwrite the item at index i
        raise NotImplementedError

    def pop(self) -> Any:
        """Remove and return the last item. Raise IndexError if empty."""
        # TODO: raise IndexError if empty; otherwise shrink logical size and return the item.
        #       (Bonus: halve the capacity when the array is <= 1/4 full to save memory.)
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"DynamicArray([{', '.join(repr(self._data[i]) for i in range(self._n))}])"
