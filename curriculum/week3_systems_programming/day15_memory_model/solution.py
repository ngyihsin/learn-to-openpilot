"""Day 15 reference solution — bump allocator."""
from __future__ import annotations


class BumpAllocator:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = capacity
        self._offset = 0
        self.high_water = 0

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def used(self) -> int:
        return self._offset

    @property
    def remaining(self) -> int:
        return self._cap - self._offset

    def alloc(self, size: int, align: int = 1) -> int:
        if size < 0:
            raise ValueError("size must be non-negative")
        if align < 1:
            raise ValueError("align must be >= 1")
        aligned = ((self._offset + align - 1) // align) * align
        end = aligned + size
        if end > self._cap:
            raise MemoryError(f"out of memory: need {end}, capacity {self._cap}")
        self._offset = end
        if end > self.high_water:
            self.high_water = end
        return aligned

    def reset(self) -> None:
        self._offset = 0
