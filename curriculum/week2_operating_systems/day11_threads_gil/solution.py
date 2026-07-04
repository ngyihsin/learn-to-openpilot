"""Day 11 reference solution — work-splitting and thread-based parallel map."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def chunk(items: list[Any], k: int) -> list[list[Any]]:
    if k <= 0:
        raise ValueError("k must be positive")
    base, rem = divmod(len(items), k)
    chunks: list[list[Any]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def threaded_sum(numbers: list[int], workers: int) -> int:
    parts = chunk(numbers, workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        partials = list(ex.map(sum, parts))
    return sum(partials)


def parallel_map(func: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(func, items))
