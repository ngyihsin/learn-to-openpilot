"""Day 21 homework — how build systems think (Make / scons).

A build system turns sources into artifacts (`main.c` -> `main.o` -> `app`) by tracking a
**dependency graph** and doing the minimum work. Two ideas power all of them:

  1. **Order:** you must build a target's prerequisites before the target itself — a
     *topological sort* of the dependency DAG (and a cycle is an error you must report).
  2. **Incrementality:** only rebuild what's stale — a target needs rebuilding if it's missing or
     any prerequisite is newer than it. That's why `make` on an unchanged tree does nothing.

openpilot uses **scons**, but the logic underneath is exactly this. You're reusing your Day 07
graph skills (topological sort = DFS; "what's affected" = reverse reachability).

Fill in every ``TODO`` and run ``pytest -q``.  `deps[target]` is the list of that target's prerequisites.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_order(deps: dict[Any, list[Any]]) -> list:
    """Return a build order: every target appears AFTER all its prerequisites. Raise ValueError
    if there's a cycle (you can't build a cyclic dependency)."""
    # TODO:
    #   - nodes = all targets plus everything appearing as a prerequisite
    #   - DFS with white/gray/black coloring; a GRAY node reached again = cycle -> ValueError
    #   - append a node to the order once all its prerequisites are done (post-order)
    #   - iterate nodes/prereqs in sorted order for a deterministic result
    raise NotImplementedError


def needs_rebuild(target: Any, deps: dict[Any, list[Any]], mtimes: dict[Any, float]) -> bool:
    """True if `target` must be (re)built: it doesn't exist yet, or any prerequisite is missing
    or newer than the target. `mtimes` maps a node to its last-modified time."""
    # TODO:
    #   - if target not in mtimes: return True   (never built)
    #   - for each prerequisite: if it's missing from mtimes, or its mtime > target's mtime -> True
    #   - otherwise False
    raise NotImplementedError


def rebuild_set(deps: dict[Any, list[Any]], changed: set) -> set:
    """Return every target that transitively depends on any file in `changed` — i.e., everything
    that must be rebuilt after those files change. (Reverse reachability in the dep graph.)"""
    # TODO:
    #   - build a reverse map: prerequisite -> {targets that depend on it}
    #   - BFS/DFS from each changed node over the reverse edges, collecting the dependents
    raise NotImplementedError
