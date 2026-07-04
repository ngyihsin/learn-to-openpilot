# Day 01 — Dynamic Arrays & Amortized Analysis

> **Week 1 · Data Structures** — the first container, and your first taste of "how fast is this, really?"

## Why today matters

Python's `list`, C++'s `std::vector`, Go's slices — they're all *dynamic arrays*. They feel
magical: you keep `.append()`-ing and they just grow. But memory doesn't grow. Under the hood
the array lives in a fixed block, and when it fills up the runtime quietly allocates a bigger
block and copies everything across. Today you build that machinery yourself and prove *why*
appending is still cheap on average.

This is also your first **amortized analysis** — the idea that an occasional expensive
operation is fine if it's rare enough. You'll use this reasoning constantly in systems work
(and openpilot's message buffers do exactly this trick).

## Learning goals

By the end you can:

- Explain the difference between an array's **size** (`len`) and its **capacity**.
- Describe why growing by *doubling* gives **amortized O(1)** append, while growing by a
  fixed amount gives O(n).
- Implement a `DynamicArray` with `append`, indexing, and `pop` from scratch.

## Do this

1. **Concept + visualization (~20 min).** Open the notebook and run every cell:
   ```bash
   jupyter lab lesson.ipynb
   ```
   You'll watch capacity double as you append, and see a plot of per-append cost spiking at
   each resize while the *average* stays flat.

2. **Homework (~60 min).** Implement the `TODO`s in `homework.py`.

3. **Grade it.**
   ```bash
   pytest -q
   # or from the repo root:  python tools/grade.py day 1
   ```
   Green means you nailed it — including the amortized-cost test, which fails loudly if you
   grow the array linearly instead of doubling.

## Hints

- Keep two numbers: `_n` (how many items you actually have) and `_cap` (how many slots you've
  allocated). They are *not* the same.
- Never read or write past index `_n - 1`, even though `_cap` slots exist.
- When full, allocate `2 * _cap` and copy. That single decision is what makes append fast.

## Check yourself

- If doubling makes appends O(1) amortized, what's the worst-case cost of a *single* append?
- Why not grow by +1 slot each time? What does that do to the total copy work over `n` appends?
- (Bonus) When should you *shrink*? Why is shrinking at half-full a bad idea? (Hint: append,
  pop, append, pop... at the boundary.)

## Where this shows up later

Day 03's ring buffer, Day 06's heap, and Week 2's OS queues are all backed by arrays. In
openpilot, the logging/messaging layer reuses fixed buffers to avoid per-message allocation —
the same "allocate big, reuse" instinct you're building today.

**Next:** Day 02 — Linked Lists (when *not* to use a contiguous array).
