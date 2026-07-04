# Day 21 — Build Systems

> **Week 3 · Systems Programming** — how `make`/`scons` decide what to (re)build. Week 3 finale.

## Why today matters

A real project is many files: sources compile to objects, objects link into a binary, and headers
are shared by many. A **build system** models this as a dependency graph and does the *minimum*
work: build prerequisites before dependents, and skip anything already up to date. That's why
`make` on an unchanged tree finishes instantly, and why editing one header can trigger a big
rebuild. openpilot uses **scons**, but the logic is universal — and it's your Day 07 graph skills
wearing a hard hat.

## Learning goals

By the end you can:

- Topologically sort a dependency DAG (prerequisites before targets) and detect cycles.
- Decide if a target is stale (missing, or a prerequisite is newer) — the heart of incremental builds.
- Compute the transitive set of targets a change forces to rebuild (reverse reachability).

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — compile a real multi-file
   C project (`.c` → `.o` → linked binary) with `gcc`, see the dependency graph, and watch an
   incremental rebuild touch only what changed.
2. **Homework (~50 min).** Implement `build_order`, `needs_rebuild`, and `rebuild_set`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 21`.

## Hints

- `build_order` is a topological sort — DFS post-order (append a node once its prerequisites are
  done) gives prerequisites first. A gray-node revisit is a cycle → raise `ValueError`.
- `needs_rebuild`: rebuild if the target is missing, or **any** prerequisite is missing or newer.
- `rebuild_set`: flip the edges (prerequisite → its dependents) and do reachability from the
  changed files.

## Check yourself

- Why must a build system topologically sort before building? What breaks if it doesn't?
- `make` uses file timestamps; some modern systems (Bazel/Buck) use content hashes instead. What's
  the advantage of hashing?
- Editing a widely-included header rebuilds the world. How do build systems keep that bearable?

## Where this shows up later

This closes Week 3. You now have the systems-programming toolkit — memory, C, syscalls, sockets,
profiling, builds — to read and build a real C/C++ project. Next comes Week 4 (PyTorch), and then
you point all of it at openpilot, which you'll clone and build with scons on Day 29.

**Next:** Week 4 — PyTorch / ExecuTorch (Day 22 already built; Day 23 next).
