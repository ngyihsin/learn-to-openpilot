"""Day 15 homework — the memory model & a bump allocator.

A process's memory has regions: the **stack** (function calls, locals — grows and shrinks
automatically) and the **heap** (dynamically allocated, you manage its lifetime). Today you
build the simplest possible heap allocator — a **bump allocator** (a.k.a. arena/linear
allocator): it hands out slices of a fixed buffer by just bumping a cursor forward. It can't
free individual allocations, only reset the whole arena — but it's blazing fast and is exactly
what real-time systems use to avoid per-frame `malloc` (openpilot allocates arenas up front for
just this reason).

You'll also handle **alignment**: hardware wants an N-byte value to start at an address that's a
multiple of N, so the allocator rounds the cursor up before carving out a block.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations


class BumpAllocator:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._cap = capacity
        self._offset = 0
        self.high_water = 0     # largest offset ever reached (survives reset)

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
        """Reserve `size` bytes aligned to `align`; return the starting offset.
        Raise MemoryError if it won't fit, ValueError for bad arguments."""
        # TODO:
        #   - validate size >= 0 and align >= 1 (else ValueError)
        #   - round the cursor up to a multiple of align:
        #       aligned = ((self._offset + align - 1) // align) * align
        #   - if aligned + size > capacity: raise MemoryError
        #   - advance the cursor, update high_water, return the aligned start offset
        raise NotImplementedError

    def reset(self) -> None:
        """Free everything at once by rewinding the cursor. high_water is preserved."""
        # TODO: set the cursor back to 0 (do NOT touch high_water)
        raise NotImplementedError
