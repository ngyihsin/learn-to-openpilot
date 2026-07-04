# Day 17 — Calling C from Python (ctypes / cffi)

> **Week 3 · Systems Programming** — cross the language boundary without leaving Python. A coding day.

## Why today matters

The reason Python can be both convenient *and* fast is that the heavy lifting happens in C: NumPy
arrays, PyTorch tensors, and openpilot's Python tooling all call into compiled code. The bridge is
a **foreign-function interface (FFI)**. Python ships with **ctypes**, which lets you load a shared
library and call its functions directly. The interesting part is **marshalling**: a Python list
isn't a C array, so you convert it, pass a pointer, and read results (or in-place mutations) back.

## Learning goals

By the end you can:

- Load a shared library and call C functions from Python via ctypes.
- Marshal a Python `list[float]` into a C `double[]` and pass it by pointer.
- Understand in-place mutation across the boundary (C wrote into *your* buffer) and read it back.

## Do this

1. **Homework (~55 min).** Implement `py_array_sum`, `py_scale`, and `py_dot` in `homework.py`.
   The library is compiled and its signatures are set for you in `_lib()`.
2. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 17`.
   *(Needs a C compiler; the grader skips if none is installed.)*

## Hints

- Build a C array from a Python list: `arr = (ctypes.c_double * n)(*values)`. Pass `arr` where a
  `double*` is expected — ctypes handles the pointer.
- `scale` mutates in place, so after the call your `arr` already holds the scaled values —
  `list(arr)` reads them back into Python.
- Always match `restype`/`argtypes` to the C signature (done for you here). Get them wrong and
  ctypes will silently misread memory — a classic FFI footgun.

## Check yourself

- Why do you have to declare `argtypes`/`restype`? What goes wrong if you don't?
- Passing a big array to C is cheap, but converting a Python list to a C array is O(n). When is
  the FFI overhead worth it, and when does it dominate?
- NumPy arrays already store a contiguous C buffer. Why does that make NumPy↔C much cheaper than
  list↔C? (This is why real code passes NumPy arrays across the boundary.)

## Where this shows up later

Day 20 profiles to decide *what* is worth pushing into C. This ctypes pattern is the toy version
of how PyTorch exposes C++/CUDA kernels — and how openpilot's Python talks to its C++ core.

**Next:** Day 18 — Syscalls & the Kernel Boundary.
