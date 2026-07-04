"""Day 07 homework — Graphs, BFS & DFS.

A graph is just nodes and edges, but it's the most *general* structure here: trees, grids, road
networks, dependency chains, and social networks are all graphs. Two traversals do most of the
work:

- **BFS** (breadth-first, a queue) explores in rings of increasing distance — so on an unweighted
  graph it finds **shortest paths**.
- **DFS** (depth-first, a stack/recursion) plunges as deep as it can before backtracking — great
  for reachability, cycle detection, and topological order.

The `Graph` container is given. Implement the four traversal functions. Run ``pytest -q``.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class Graph:
    """Adjacency-list graph. Undirected by default. `neighbors` returns a *sorted* list so
    traversal orders are deterministic (and gradeable)."""

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


# ----- implement below -----

def bfs(graph: Graph, start: Any) -> list:
    """Return nodes in breadth-first order from `start`."""
    # TODO: queue (deque) + visited set; pop from the left, append unvisited neighbors
    raise NotImplementedError


def dfs(graph: Graph, start: Any) -> list:
    """Return nodes in depth-first pre-order from `start`."""
    # TODO: recurse (or use an explicit stack); visit a node, then recurse into unvisited neighbors
    raise NotImplementedError


def shortest_path(graph: Graph, start: Any, goal: Any) -> list | None:
    """Return a shortest node path from start to goal (inclusive), or None if unreachable.
    BFS finds it because it reaches every node by the fewest edges."""
    # TODO: BFS while recording each node's parent; when you dequeue/discover goal, walk the
    #       parents back to start and reverse. start == goal -> [start].
    raise NotImplementedError


def grid_shortest_path(grid: list[str], start: tuple[int, int],
                       goal: tuple[int, int]) -> list | None:
    """Shortest 4-directional path on a character grid where '#' is a wall and anything else is
    walkable. Return the list of (row, col) cells from start to goal, or None if blocked."""
    # TODO: BFS over grid cells; a cell is walkable if it's in bounds and grid[r][c] != '#'.
    #       Move up/down/left/right. Reconstruct the path from parents like shortest_path.
    raise NotImplementedError
