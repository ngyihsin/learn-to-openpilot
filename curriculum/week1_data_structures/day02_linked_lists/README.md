# Day 02 — Linked Lists

> **Week 1 · Data Structures** — when *not* to use a contiguous array.

## Why today matters

Yesterday's dynamic array is great for random access (`a[i]` is instant) but terrible for
inserting in the middle — everything after the gap has to shift over. A **linked list** makes
the opposite trade: it threads items together with pointers, so inserting or deleting at a
node you already hold is O(1) (just rewire a couple of pointers), but reaching the *i*-th item
means walking there.

Knowing which trade-off you want — contiguous vs. linked — is one of the most practical
instincts in systems programming. Queues, LRU caches (you'll build one on Day 10!), and kernel
free-lists are all linked structures.

## Learning goals

By the end you can:

- Explain the pointer/`prev`/`next` structure of a node and why *doubly*-linked enables O(1)
  deletion and backward traversal.
- Implement `push_front/back`, `pop_front/back`, `delete_value`, and in-place `reverse`.
- Handle the boundary cases that cause 90% of linked-list bugs: the empty list, and operations
  at the head or tail (where a neighbour pointer is `None`).

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — you'll *see* nodes as
   boxes and arrows, and watch an insert rewire exactly two pointers while the array version
   shifts a whole tail.
2. **Homework (~60 min).** Implement the `TODO`s in `homework.py`. Traversal helpers are given.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 2`.

## Hints

- Every method has two special cases: acting on an **empty** list, and acting at an **end**
  (where `prev` or `next` is `None`). When a neighbour is `None`, you're updating `head`/`tail`
  instead of that neighbour.
- The grader checks `to_list()` *and* `to_list_reverse()` after every op. If forward looks
  right but backward is wrong, you dropped a `prev` link somewhere.
- For `reverse`, swapping each node's `prev`/`next` means "forward" is now via `prev` — that's
  why the walk advances with `cur = cur.prev`.

## Check yourself

- Inserting at the front of a 1-million-item list: O(1) for a linked list, O(n) for a dynamic
  array. Why? What about `a[500000]`?
- Why does a *singly*-linked list make O(1) deletion of an arbitrary node hard, but a doubly
  -linked list makes it easy?
- Where would a real system prefer a linked list over an array? (Hint: a scheduler's run queue.)

## Where this shows up later

Day 03 builds stacks & queues (often on linked lists), and Day 10's LRU page-replacement cache
is essentially a linked list + hash map. In openpilot, message and buffer bookkeeping lean on
these O(1) enqueue/dequeue patterns.

**Next:** Day 03 — Stacks & Queues.
