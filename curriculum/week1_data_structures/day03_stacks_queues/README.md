# Day 03 — Stacks, Queues & Ring Buffers

> **Week 1 · Data Structures** — three tiny containers you'll use everywhere. A pure coding day: no notebook, just build and test.

## Why today matters

- A **stack** (LIFO) *is* how function calls work — and how you'll do depth-first search on Day 07.
- A **queue** (FIFO) is how work and messages flow through a system — and how breadth-first search works.
- A **ring buffer** is a fixed-capacity queue in reused memory. Real-time systems love it because
  it never allocates mid-stream and can't grow unbounded. openpilot's message plumbing is full
  of this pattern.

## Learning goals

By the end you can:

- Implement stack and queue ADTs and state their LIFO/FIFO guarantees.
- Build a queue from **two stacks** and explain why dequeue is *amortized* O(1).
- Implement a circular (ring) buffer with head + count indices and wraparound arithmetic.

## Do this

1. **Homework (~60 min).** Implement `Stack`, `Queue`, and `RingBuffer` in `homework.py`.
2. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 3`.

## Hints

- **Queue-from-two-stacks:** push onto `_in`. To dequeue, if `_out` is empty, pour `_in` into
  `_out` (popping from one and pushing to the other *reverses* the order), then pop `_out`. Each
  item is moved at most once → amortized O(1).
- **Ring buffer:** don't track a tail index — compute it as `(head + count) % capacity`. Push
  writes there and bumps `count`; pop reads at `head` and advances `head = (head+1) % capacity`.
  The grader's wraparound test fails if your modular arithmetic is off.

## Check yourself

- Why is a single dequeue occasionally O(n) but the *average* still O(1)? (Sound familiar? It's
  the same amortized argument as Day 01's doubling array.)
- What breaks if a ring buffer tracks a tail index but forgets to distinguish "empty" from
  "full" (both can look like head == tail)? (That's why we track `count`.)
- When would you *want* a bounded buffer to reject new items vs. overwrite the oldest?

## Where this shows up later

Day 06's heap is a priority queue; Day 07 uses a stack for DFS and a queue for BFS. Week 2's
schedulers and IPC channels are queues. The ring buffer returns in any real-time/streaming context.

**Next:** Day 04 — Hash Maps.
