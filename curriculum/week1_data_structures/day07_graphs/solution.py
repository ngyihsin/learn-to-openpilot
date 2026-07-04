"""Day 07 reference solution — Graph traversals & shortest paths."""
from __future__ import annotations

from collections import deque
from typing import Any


class Graph:
    def __init__(self, directed: bool = False) -> None:
        self._adj: dict[Any, set] = {}
        self._directed = directed

    def add_node(self, u: Any) -> None:
        self._adj.setdefault(u, set())

    def add_edge(self, u: Any, v: Any) -> None:
        self.add_node(u)
        self.add_node(v)
        self._adj[u].add(v)
        if not self._directed:
            self._adj[v].add(u)

    def neighbors(self, u: Any) -> list:
        return sorted(self._adj.get(u, set()))

    def nodes(self) -> list:
        return sorted(self._adj)


def bfs(graph: Graph, start: Any) -> list:
    visited = {start}
    order: list = []
    q = deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return order


def dfs(graph: Graph, start: Any) -> list:
    visited: set = set()
    order: list = []

    def walk(u: Any) -> None:
        visited.add(u)
        order.append(u)
        for v in graph.neighbors(u):
            if v not in visited:
                walk(v)

    walk(start)
    return order


def _reconstruct(parent: dict, goal: Any) -> list:
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def shortest_path(graph: Graph, start: Any, goal: Any) -> list | None:
    if start == goal:
        return [start]
    visited = {start}
    parent = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                if v == goal:
                    return _reconstruct(parent, goal)
                q.append(v)
    return None


def grid_shortest_path(grid: list[str], start: tuple[int, int],
                       goal: tuple[int, int]) -> list | None:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def walkable(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "#"

    if not walkable(*start) or not walkable(*goal):
        return None
    if start == goal:
        return [start]

    visited = {start}
    parent: dict = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = (r + dr, c + dc)
            if walkable(*cell) and cell not in visited:
                visited.add(cell)
                parent[cell] = (r, c)
                if cell == goal:
                    return _reconstruct(parent, goal)
                q.append(cell)
    return None
