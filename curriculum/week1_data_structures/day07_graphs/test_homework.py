"""Auto-grader for Day 07 — Graphs, BFS & DFS.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _load(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load(os.environ.get("LP_IMPL", "homework"))
Graph = _impl.Graph
bfs, dfs, shortest_path, grid_shortest_path = (
    _impl.bfs, _impl.dfs, _impl.shortest_path, _impl.grid_shortest_path,
)


def make_graph() -> "Graph":
    g = Graph()
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]:
        g.add_edge(u, v)
    return g


def test_bfs_order():
    # Rings from 0: {0} then {1,2} then {3} then {4}. Ties broken by sorted neighbors.
    assert bfs(make_graph(), 0) == [0, 1, 2, 3, 4]


def test_dfs_order():
    # Depth-first with sorted neighbors: 0 -> 1 -> 3 -> 2 (dead end) -> 4.
    assert dfs(make_graph(), 0) == [0, 1, 3, 2, 4]


def test_bfs_visits_all_reachable_only():
    g = make_graph()
    g.add_node(99)                    # isolated
    assert set(bfs(g, 0)) == {0, 1, 2, 3, 4}


def test_shortest_path_is_fewest_edges():
    assert shortest_path(make_graph(), 0, 4) == [0, 1, 3, 4]   # 3 edges, not via 2


def test_shortest_path_same_node():
    assert shortest_path(make_graph(), 2, 2) == [2]


def test_shortest_path_unreachable():
    g = make_graph()
    g.add_node(99)
    assert shortest_path(g, 0, 99) is None


def test_grid_path_around_wall():
    grid = [
        ".....",
        ".###.",
        ".....",
    ]
    path = grid_shortest_path(grid, (0, 0), (2, 4))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (2, 4)
    assert len(path) == 7             # 6 steps is the shortest way around the wall
    # Every step is a single 4-directional move onto a non-wall cell.
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert grid[r2][c2] != "#"


def test_grid_no_path():
    grid = [
        "...",
        "###",
        "..G",   # 'G' is walkable (not '#'), but row of walls seals it off
    ]
    assert grid_shortest_path(grid, (0, 0), (2, 2)) is None


def test_grid_goal_is_wall():
    grid = ["..", ".#"]
    assert grid_shortest_path(grid, (0, 0), (1, 1)) is None
