# Day 06 — Heaps & Priority Queues

> **Week 1 · Data Structures** — always know the smallest thing, in O(log n).

## Why today matters

A priority queue answers "what's the most urgent item right now?" over and over, efficiently.
The standard implementation is a **binary heap**: a complete binary tree packed into a plain
array, where each parent is ≤ its children. The minimum is always at index 0. It powers
Dijkstra and A* pathfinding (Day 07!), event-driven simulation, Huffman coding, and the
shortest-job-first scheduler you'll build on Day 08.

The elegant part: no pointers, no nodes — just index arithmetic. Node `i`'s children are `2i+1`
and `2i+2`; its parent is `(i-1)//2`.

## Learning goals

By the end you can:

- Explain the array/complete-tree layout and the heap-order invariant.
- Implement `_sift_up` and `_sift_down`, and build `push`/`pop` on top of them.
- Build a heap from a list in O(n) with `heapify`, and use pop-until-empty as **heapsort**.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — see the array drawn as a
   tree, watch a pushed value sift up to its place, and watch heapsort drain the heap into
   sorted order.
2. **Homework (~60 min).** Implement `_sift_up`, `_sift_down`, `push`, `pop`, and `heapify`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 6`.

## Hints

- **`_sift_up`:** compare with parent `(i-1)//2`; swap upward while you're smaller; stop at the root.
- **`_sift_down`:** find the *smaller* of the two children; if it beats the current node, swap and
  continue there; otherwise you're done.
- **`pop`:** swap the last element into position 0, shrink the array, then sift that element down.
  Watch the single-element and empty cases.
- **`heapify`:** sifting down from the last parent (`len//2 - 1`) back to the root builds a valid
  heap in O(n) — cheaper than n separate pushes.

## Check yourself

- Why is a heap stored in a flat array with no pointers, unlike Day 05's BST?
- A heap gives O(1) *min* but not O(log n) *search for an arbitrary value*. Why? When does that
  trade-off make a heap the right choice over a BST?
- Heapsort pops n times, each O(log n) → O(n log n), in place. What's it giving up vs. quicksort?

## Where this shows up later

Day 07 uses a priority queue for weighted shortest paths; Day 08's SJF scheduler repeatedly picks
the smallest remaining job — a min-heap by burst time. Priority queues are ubiquitous in
schedulers and planners, including motion/route planning.

**Next:** Day 07 — Graphs, BFS & DFS.
