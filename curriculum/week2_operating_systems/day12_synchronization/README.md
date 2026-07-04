# Day 12 — Synchronization: Locks & Semaphores

> **Week 2 · Operating Systems** — making shared mutable state safe.

## Why today matters

The moment two threads touch the same variable and at least one writes, you have a **race
condition**. `count += 1` looks atomic but isn't — it's read, add, write — so two threads can
both read the same old value and one increment vanishes. The fix is **mutual exclusion**: a
**lock** that only one thread can hold at a time. For "wait until something is true" (buffer not
full, work available), you use a **condition variable** so a thread sleeps instead of busy-waiting.

The **bounded buffer** (producer/consumer) you'll build is the beating heart of every task
queue, thread pool, and message pipeline — including the ones moving sensor data between
processes in a robotics stack.

## Learning goals

By the end you can:

- Explain why `x += 1` across threads loses updates, and fix it with a lock.
- Use a condition variable to block until a predicate holds (and why it's a `while`, not an `if`).
- Implement a thread-safe bounded buffer with correct `put`/`get` blocking.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — watch an unlocked counter
   *lose* updates (final value < expected) across runs, then watch the lock make it exact.
2. **Homework (~60 min).** Implement `SafeCounter.increment` and `BoundedBuffer.put/get`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 12`.

## Hints

- Wrap the read-modify-write in `with self._lock:` — the `with` guarantees the lock is released
  even if something throws.
- In `put`/`get`, wait in a **`while`** loop, not an `if`: when a thread wakes, the condition it
  waited for may already have been snatched by another thread (spurious/stolen wakeups). Re-check.
- `notify_all()` after changing the buffer so blocked producers *and* consumers get a chance to
  proceed.

## Check yourself

- Exactly how does `x += 1` lose an update? Write out the interleaving of two threads.
- Why must the wait be a `while` loop? What bug appears if you use `if`?
- A lock gives *mutual exclusion*; a **semaphore** counts permits (e.g. "at most N in the pool").
  How would you cap concurrent downloads to 5 with a semaphore?

## Where this shows up later

Day 13 shows how locks themselves go wrong — **deadlock** — and how to prevent it. Day 14 moves
to processes that can't share memory at all and must pass messages. Every real concurrent
system, openpilot included, is built on these primitives.

**Next:** Day 13 — Deadlock & Race Conditions.
