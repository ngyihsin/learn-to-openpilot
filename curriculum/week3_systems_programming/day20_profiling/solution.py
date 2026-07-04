"""Day 20 reference solution — linear-time versions."""
from __future__ import annotations

from typing import Any


def has_duplicate(items: list[Any]) -> bool:
    seen: set = set()
    for x in items:
        if x in seen:
            return True
        seen.add(x)
    return False


def common_elements(a: list[Any], b: list[Any]) -> set:
    return set(a) & set(b)
