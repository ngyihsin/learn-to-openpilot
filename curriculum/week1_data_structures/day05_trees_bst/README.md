# Day 05 — Trees & Binary Search Trees

> **Week 1 · Data Structures** — ordered data with O(log n) search... when you keep it balanced.

## Why today matters

A binary search tree keeps values in sorted structure: smaller to the left, larger to the
right. That single invariant buys you O(height) search, insert, and delete, and gives you sorted
output for free via an in-order walk. Trees are everywhere underneath: database indexes,
filesystem directories, expression parsers, and the balanced-tree variants (red-black, B-trees)
that power real storage engines.

You'll also feel the catch: insert already-sorted data and the tree degenerates into a linked
list (height n−1), and your "O(log n)" becomes O(n). That failure mode is *why* balanced trees exist.

## Learning goals

By the end you can:

- State and maintain the BST invariant through insert and delete.
- Implement the three traversals and know that **in-order = sorted** for a BST.
- Implement the three-case `delete` (leaf / one child / two children via in-order successor).
- Explain why an unbalanced BST loses its performance guarantee.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — draw a tree, watch an
   in-order walk emit sorted values, then watch a *sorted* insert order collapse the tree into a
   spindly chain (height blows up).
2. **Homework (~70 min).** Implement `insert`, `contains`, `in_order`, and `delete`. Traversal
   and min/max helpers are provided.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 5`.

## Hints

- Recursion mirrors the structure: `insert`/`delete` return the (possibly new) subtree root, and
  the caller reattaches it — `node.left = dele(node.left, value)`. This "rebuild the spine on the
  way back up" pattern makes deletion clean.
- **Delete's two-children case:** copy in the in-order successor (the smallest value in the right
  subtree, which `_min_node` finds), then delete that successor from the right subtree.
- The grader checks `in_order()` stays sorted after every delete — that's the invariant.

## Check yourself

- Why does an in-order traversal of a BST come out sorted?
- Insert `1,2,3,4,5` in that order: what's the height, and why is that bad? What insert order
  gives a balanced tree?
- Why is the in-order *successor* always guaranteed to have at most one child (so deleting it is easy)?

## Where this shows up later

Day 06's heap is a different tree (a *complete* binary tree in an array). Balanced trees and
B-trees underpin the indexes you'll meet in any database. The recursion patterns here return in
Day 07's graph search.

**Next:** Day 06 — Heaps & Priority Queues.
