"""Day 13 reference solution — deadlock detection (wait-for-graph cycle finding)."""
from __future__ import annotations

from typing import Any

_WHITE, _GRAY, _BLACK = 0, 1, 2


def find_deadlock_cycle(wait_for: dict[Any, list[Any]]) -> list | None:
    nodes = set(wait_for) | {v for vs in wait_for.values() for v in vs}
    color = {u: _WHITE for u in nodes}
    path: list[Any] = []

    def dfs(u: Any) -> list | None:
        color[u] = _GRAY
        path.append(u)
        for v in sorted(wait_for.get(u, []), key=str):
            if color[v] == _GRAY:
                return path[path.index(v):]          # back-edge -> cycle
            if color[v] == _WHITE:
                found = dfs(v)
                if found is not None:
                    return found
        color[u] = _BLACK
        path.pop()
        return None

    for u in sorted(nodes, key=str):
        if color[u] == _WHITE:
            found = dfs(u)
            if found is not None:
                return found
    return None


def has_deadlock(wait_for: dict[Any, list[Any]]) -> bool:
    return find_deadlock_cycle(wait_for) is not None


def safe_lock_order(locks: list[Any]) -> list[Any]:
    return sorted(locks, key=str)
