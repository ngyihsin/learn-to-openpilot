# Day 13 — Deadlock & Race Conditions

> **Week 2 · Operating Systems** — when locks, meant to keep you safe, freeze everything.

## Why today matters

Locks (Day 12) prevent races, but misuse them and you get the opposite disaster: **deadlock**,
where threads wait on each other in a cycle and nobody ever proceeds. The textbook case: thread
A grabs lock 1 then wants lock 2; thread B grabs lock 2 then wants lock 1. Both wait forever.

The beautiful part: draw "who waits for whom" as a directed graph — the **wait-for graph** — and
a deadlock is *exactly a cycle*. So detecting deadlock is the cycle-detection you already wrote
on Day 07. And preventing it is often one rule: **always acquire locks in a consistent global
order**, which makes a cycle impossible.

## Learning goals

By the end you can:

- State the four Coffman conditions and, concretely, how a wait-for cycle forms.
- Detect deadlock by finding a cycle in the wait-for graph (DFS with white/gray/black coloring).
- Prevent deadlock with a global lock ordering, and explain *why* it works.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — reproduce a real data
   race (lost updates), then a real two-lock deadlock (demonstrated safely with lock timeouts),
   then fix it by acquiring locks in sorted order. See the wait-for graph and its cycle.
2. **Homework (~55 min).** Implement `find_deadlock_cycle`, `has_deadlock`, and `safe_lock_order`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 13`.

## Hints

- **Cycle detection via coloring:** WHITE = unvisited, GRAY = on the current DFS path, BLACK =
  fully explored. An edge to a **GRAY** node is a back-edge — a cycle. The cycle is the slice of
  your path stack from that gray node to the current node.
- Include nodes that appear *only* as values (things being waited on), not just dict keys.
- `safe_lock_order` just sorts by a stable key — the point is that *every* thread sorts the same
  way, so their acquisition orders can never form a cycle.

## Check yourself

- The four conditions for deadlock are mutual exclusion, hold-and-wait, no preemption, and
  circular wait. Which one does a global lock order eliminate?
- Why does a self-loop in the wait-for graph (A waits for A) count as a deadlock?
- Detection vs. prevention vs. avoidance (banker's algorithm): when would you choose each?

## Where this shows up later

Day 14 sidesteps shared-lock deadlocks by using message passing between processes. In any
lock-heavy system — databases, kernels, robotics middleware — "acquire in a fixed order" is the
rule that keeps you out of trouble.

**Next:** Day 14 — IPC, Pipes & the File System.
