# Day 11 — Threads & the GIL

> **Week 2 · Operating Systems** — shared-memory parallelism, and why Python threads can't speed up math.

## Why today matters

Threads live *inside* a process and share its memory, so they can cooperate without copying data
— but that sharing is exactly what makes concurrency bugs possible (that's Days 12–13). CPython
adds a famous catch: the **Global Interpreter Lock** lets only one thread execute Python
bytecode at a time. So threads give you **no speedup on CPU-bound** work — but they're still
great for **I/O-bound** work (waiting on disk/network releases the GIL), and **processes** avoid
the GIL entirely by not sharing an interpreter.

Knowing which tool fits — thread vs. process vs. async — is one of the most practical
performance decisions you'll make. openpilot runs work across **processes** for exactly this reason.

## Learning goals

By the end you can:

- Explain the difference between threads and processes, and what the GIL does.
- Split work into balanced chunks and run a map across a thread pool (order-preserving).
- Predict when threads help (I/O-bound) vs. when you need processes (CPU-bound).

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — benchmark the same
   CPU-bound work three ways (serial, threads, processes) and watch threads come out **no faster
   than serial** while processes scale. That bar chart is the GIL, made visible.
2. **Homework (~50 min).** Implement `chunk`, `threaded_sum`, and `parallel_map`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 11`.

## Hints

- `chunk`: `base, rem = divmod(len(items), k)`; the first `rem` chunks get one extra element.
  Concatenating the chunks must reproduce the input exactly.
- These functions are thread-safe *by design*: each worker owns a disjoint slice and returns its
  own partial result, so there's no shared mutable state to corrupt. (Contrast that with Day 12.)
- `ThreadPoolExecutor.map` preserves input order even when tasks finish out of order — the grader
  checks that with a deliberately uneven-timing test.

## Check yourself

- Why does `threaded_sum` give no speedup for pure arithmetic in CPython, but downloading 100
  URLs with threads *is* much faster?
- Threads share memory; processes don't. What's the cost of using processes instead? (Hint:
  data has to be *serialized* to cross the boundary — that's Day 14.)
- Why is "each thread owns its own slice" the easiest way to write correct concurrent code?

## Where this shows up later

Day 12 tackles the hard case — threads that *must* share mutable state — with locks and
semaphores. Day 13 shows how sharing goes wrong (races, deadlock). Day 14 covers processes
talking via pipes. openpilot's multi-process design is the real-world payoff.

**Next:** Day 12 — Synchronization: Locks & Semaphores.
