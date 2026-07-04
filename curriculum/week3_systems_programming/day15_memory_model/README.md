# Day 15 — Memory Model: Stack, Heap & Pointers

> **Week 3 · Systems Programming** — where your data actually lives, and who cleans it up.

## Why today matters

In Python, memory "just works." In C — and in the C++ core of openpilot — you manage it
yourself, so you have to know the terrain: the **stack** (automatic, LIFO, holds call frames and
locals, freed when a function returns) vs. the **heap** (you allocate, you free, lives as long as
you want). Getting this wrong is where segfaults, leaks, and use-after-free bugs come from.

You'll build the simplest real allocator — a **bump allocator** — which carves slices off a fixed
buffer by advancing a cursor. It can't free one allocation at a time (only reset the whole
arena), but it's extremely fast and allocation-free at steady state, which is exactly why
real-time systems pre-allocate arenas instead of calling `malloc` in the hot loop.

## Learning goals

By the end you can:

- Describe the stack vs. heap and what "a pointer is just an address" means.
- Implement a bump/arena allocator with correct **alignment** and out-of-memory handling.
- Explain why arena allocation is fast and where it fits (and doesn't).

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — see a process's address
   space (stack growing down, heap up), and watch the bump cursor march forward with alignment
   padding between blocks.
2. **Homework (~50 min).** Implement `alloc` (with alignment + OOM) and `reset`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 15`.

## Hints

- **Alignment:** round the cursor up before carving: `((offset + align - 1) // align) * align`.
  A failed (OOM) allocation must **not** advance the cursor.
- `reset` rewinds the cursor to 0 but keeps `high_water` — that peak tells you how big the arena
  really needs to be.

## Check yourself

- Why does `align=8` for an 8-byte double sometimes waste a few bytes of padding? Why does the
  hardware care?
- A bump allocator can't free individual objects. What kinds of workloads make that totally fine
  (hint: "do a batch of work, then throw it all away")?
- What's the difference between a dangling pointer and a memory leak?

## Where this shows up later

Day 16–17 write and call real C where *you* own the memory. Day 20 profiles allocation-heavy
code. Arena allocation is a staple of game engines, compilers, and real-time robotics loops.

**Next:** Day 16 — C for Python Programmers.
