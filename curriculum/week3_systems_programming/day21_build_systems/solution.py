"""Day 21 reference solution — build ordering & incremental rebuild logic."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

_WHITE, _GRAY, _BLACK = 0, 1, 2


def build_order(deps: dict[Any, list[Any]]) -> list:
    nodes = set(deps) | {p for ps in deps.values() for p in ps}
    color = {n: _WHITE for n in nodes}
    order: list = []

    def visit(n: Any) -> None:
        color[n] = _GRAY
        for p in sorted(deps.get(n, []), key=str):
            if color[p] == _GRAY:
                raise ValueError(f"dependency cycle at {p!r}")
            if color[p] == _WHITE:
                visit(p)
        color[n] = _BLACK
        order.append(n)          # post-order: prerequisites land before dependents

    for n in sorted(nodes, key=str):
        if color[n] == _WHITE:
            visit(n)
    return order


def needs_rebuild(target: Any, deps: dict[Any, list[Any]], mtimes: dict[Any, float]) -> bool:
    if target not in mtimes:
        return True
    target_time = mtimes[target]
    for prereq in deps.get(target, []):
        if prereq not in mtimes or mtimes[prereq] > target_time:
            return True
    return False


def rebuild_set(deps: dict[Any, list[Any]], changed: set) -> set:
    reverse: dict[Any, set] = defaultdict(set)
    for target, prereqs in deps.items():
        for p in prereqs:
            reverse[p].add(target)

    result: set = set()
    stack = list(changed)
    while stack:
        node = stack.pop()
        for dependent in reverse.get(node, ()):
            if dependent not in result:
                result.add(dependent)
                stack.append(dependent)
    return result
