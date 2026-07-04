"""Day 13 homework — Deadlock detection & prevention.

A **deadlock** is when a set of threads/processes are all stuck waiting for each other forever:
A holds lock 1 and wants lock 2, while B holds lock 2 and wants lock 1. Draw "who is waiting for
whom" as a directed graph — the **wait-for graph** — and a deadlock is exactly a **cycle** in it.

So detecting deadlock reuses your Day 07 graph skills: find a cycle. Preventing it is often as
simple as a rule: always acquire locks in a single, consistent global order (no cycle can form).

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from typing import Any


def find_deadlock_cycle(wait_for: dict[Any, list[Any]]) -> list | None:
    """Return one cycle in the wait-for graph (as an ordered list of nodes), or None if the
    graph is acyclic. `wait_for[x]` is the list of nodes x is currently blocked waiting on.

    Use a depth-first search that colors nodes white/gray/black: reaching a **gray** node (one
    still on the current DFS path) means you've found a back-edge — a cycle. Return the path
    from that gray node to the current node.
    """
    # TODO:
    #   - collect all nodes (keys AND anything that appears in a value list)
    #   - DFS with a WHITE/GRAY/BLACK coloring and a path stack
    #   - when an edge leads to a GRAY node v, return path_stack[index_of(v):]
    #   - iterate start nodes and neighbors in sorted order for deterministic results
    raise NotImplementedError


def has_deadlock(wait_for: dict[Any, list[Any]]) -> bool:
    """True iff the wait-for graph contains a cycle."""
    # TODO: return find_deadlock_cycle(wait_for) is not None
    raise NotImplementedError


def safe_lock_order(locks: list[Any]) -> list[Any]:
    """Return `locks` in a consistent global order (sorted by str(lock)). Acquiring locks in the
    SAME order everywhere makes a wait-for cycle — and thus deadlock — impossible."""
    # TODO: return the locks sorted by a stable key
    raise NotImplementedError
