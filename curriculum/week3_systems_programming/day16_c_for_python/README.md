# Day 16 — C for Python Programmers

> **Week 3 · Systems Programming** — the language the machine actually runs. A pure coding day.

## Why today matters

openpilot's performance-critical core is C++. NumPy, PyTorch, CPython itself — all C underneath.
You don't need to become a C wizard, but you need to *read* it and write small, correct functions
in it. The good news: if you know Python loops and conditionals, C is mostly the same ideas with
explicit types and manual care about size/overflow. Today you port three small "hot" functions to
C and prove they match your Python — the exact skill behind "rewrite this hot loop in C for speed."

## Learning goals

By the end you can:

- Read and write basic C: typed variables, `for` loops, `if`, integer types, and return values.
- Explain why integer type/size matters (overflow) in a way Python hides from you.
- Compile C into a shared library and call it from Python (previewing Day 17's FFI).

## Do this

1. **Homework (~60 min).** Implement `sum_squares`, `is_prime`, and `fib` in **`hot.c`**.
2. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 16`. The grader compiles your C
   with `gcc` and calls it via ctypes, comparing against a Python reference.
   *(Needs a C compiler; the grader skips if none is installed.)*

## Hints

- Declare a type for every variable: `long long sum = 0;`. A sum of squares up to n=100000 blows
  past 32-bit `int` — that's why the return types are `long long`. Watch the cast: `(long long)i * i`.
- `for (long i = 0; i < n; i++) { ... }` — semicolons and braces, no colons or indentation rules.
- `fib` must be **iterative** (two running values). Recursive Fibonacci is exponential; the grader
  tests `fib(90)`, which recursion would never finish.

## Check yourself

- Why does `int` overflow silently in C but never in Python? What does that buy C in speed?
- In `is_prime`, why check divisors only up to `i*i <= n` instead of `i <= n`?
- What does compiling to a `.so` (shared object) give you that a standalone program doesn't?

## Where this shows up later

Day 17 wraps richer C (arrays, in-place mutation, structs) from Python with ctypes and measures
the speedup. Day 20 profiles to *find* the hot function worth porting. Day 21 builds multi-file C
projects the way openpilot's build system does.

**Next:** Day 17 — Calling C from Python (ctypes / cffi).
