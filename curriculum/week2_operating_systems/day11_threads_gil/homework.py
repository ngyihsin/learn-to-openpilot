"""Day 11 homework — Threads, work-splitting, and the GIL.

Threads share a process's memory, which makes them cheap to communicate through but dangerous
to get wrong (Days 12–13). In CPython there's an extra twist: the **Global Interpreter Lock**
means only one thread runs Python bytecode at a time, so threads *don't* speed up CPU-bound work
— but they're great for I/O-bound work, and processes sidestep the GIL entirely. You'll *see*
that in the notebook benchmark.

Here you'll build the plumbing that both threads and processes need: splitting work into chunks
and running a map in parallel. These functions are deterministic and thread-safe by design —
each worker owns its own slice, so there's nothing to race on.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def chunk(items: list[Any], k: int) -> list[list[Any]]:
    """Split `items` into exactly `k` contiguous chunks whose sizes differ by at most 1.

    e.g. chunk([0..9], 3) -> [[0,1,2,3],[4,5,6],[7,8,9]].  Concatenating the chunks must
    reproduce `items`. If k > len(items), the extra chunks are empty lists.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    # TODO:
    #   base, rem = divmod(len(items), k)
    #   the first `rem` chunks get base+1 items, the rest get base; walk a cursor through items
    raise NotImplementedError


def threaded_sum(numbers: list[int], workers: int) -> int:
    """Sum `numbers` by splitting into `workers` chunks, summing each chunk in its own thread,
    then combining. The result must equal the plain sum — this is safe because each thread
    reads a disjoint slice and returns its own partial sum (no shared mutable state)."""
    # TODO:
    #   - split with chunk(numbers, workers)
    #   - run sum() on each chunk in a ThreadPoolExecutor
    #   - return the sum of the partial sums
    raise NotImplementedError


def parallel_map(func: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """Apply `func` to every item across `workers` threads, returning results in the SAME order
    as `items` (even though threads may finish out of order)."""
    # TODO: use a ThreadPoolExecutor; executor.map preserves input order — or submit and
    #       collect by index yourself.
    raise NotImplementedError
