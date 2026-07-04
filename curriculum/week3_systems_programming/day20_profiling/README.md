# Day 20 — Profiling & Performance

> **Week 3 · Systems Programming** — measure first, then make it fast.

## Why today matters

The cardinal sin of optimization is guessing. Real speedups come from **profiling** to find where
time actually goes, then fixing *that* — and the fix is usually algorithmic (delete a nested
loop), not micro-tuning. Today you take two correct-but-quadratic functions, and turn O(n²) into
O(n) with the right data structure. This "spot the accidental nested loop" instinct is one of the
highest-leverage skills in all of engineering.

## Learning goals

By the end you can:

- Use `timeit`/`cProfile` to find a hot spot instead of guessing.
- Recognize accidental O(n²) (a lookup inside a loop) and fix it with a set/dict → O(n).
- Reason about time budgets: why "instant on 100 items" can be "10 seconds on 50,000."

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — profile a slow function,
   see the hot line, fix it, and plot naive-vs-fast runtime as input size grows (the curves
   diverge dramatically).
2. **Homework (~40 min).** Rewrite `has_duplicate` and `common_elements` to be linear.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 20`. The grader checks correctness
   *and* runs your code on 50,000 items with a time budget — an O(n²) answer will time out.

## Hints

- Membership tests: `x in a_list` is O(n), but `x in a_set` is O(1). Swapping a list for a set
  inside a loop is the whole game.
- `has_duplicate`: remember what you've seen in a set; return early on the first repeat.
- `common_elements`: `set(a) & set(b)`.

## Check yourself

- Why is `x in some_list` slow but `x in some_set` fast? (Day 04 — hashing!)
- Big-O hides constants. When might an O(n²) solution actually beat an O(n) one? (Tiny n, or huge
  constant factors / memory effects.)
- What does a profiler tell you that `time.time()` around the whole program doesn't?

## Where this shows up later

Day 21's incremental builds are a performance idea (don't rebuild what didn't change). In Week 4,
the same "profile then fix" loop applies to model latency (Day 28's quantization). openpilot lives
or dies by hitting per-frame time budgets.

**Next:** Day 21 — Build Systems.
