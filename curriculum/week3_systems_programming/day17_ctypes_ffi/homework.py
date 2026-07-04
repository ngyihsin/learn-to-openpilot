"""Day 17 homework — calling C from Python with ctypes.

Yesterday you wrote C. Today you call it from Python without leaving the interpreter, using
**ctypes** (Python's built-in foreign-function interface). The real skill isn't calling the
function — it's **marshalling data across the boundary**: turning a Python list into a C array
of doubles, passing a pointer, and reading results (or mutated buffers) back. This is exactly
how NumPy, PyTorch, and openpilot's Python tooling reach into C/C++.

The shared library is compiled and its signatures are configured for you in ``_lib()``. Your job
is the three wrappers that convert between Python and C data. Run ``pytest -q``.

C signatures available on the lib:
    double array_sum(const double *a, int n)
    void   scale(double *a, int n, double k)          # mutates the array in place
    double dot(const double *a, const double *b, int n)
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile

_LIB = None


def _lib() -> ctypes.CDLL:
    """Compile native.c (once) and return the loaded library with signatures set."""
    global _LIB
    if _LIB is not None:
        return _LIB
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler available")
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native.c")
    so = os.path.join(tempfile.mkdtemp(), "native.so")
    subprocess.run([cc, "-shared", "-fPIC", "-O2", src, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    dptr = ctypes.POINTER(ctypes.c_double)
    lib.array_sum.restype = ctypes.c_double
    lib.array_sum.argtypes = [dptr, ctypes.c_int]
    lib.scale.restype = None
    lib.scale.argtypes = [dptr, ctypes.c_int, ctypes.c_double]
    lib.dot.restype = ctypes.c_double
    lib.dot.argtypes = [dptr, dptr, ctypes.c_int]
    _LIB = lib
    return _LIB


def py_array_sum(values: list[float]) -> float:
    """Marshal `values` into a C double array and return array_sum(...)."""
    # TODO:
    #   n = len(values)
    #   arr = (ctypes.c_double * n)(*values)   # a C array initialized from the Python list
    #   return _lib().array_sum(arr, n)
    raise NotImplementedError


def py_scale(values: list[float], k: float) -> list[float]:
    """Scale `values` by k using the in-place C `scale`, and return the result as a Python list.
    (Demonstrates that C mutated the very buffer you passed.)"""
    # TODO: build the C array, call _lib().scale(arr, n, k), then return list(arr)
    raise NotImplementedError


def py_dot(a: list[float], b: list[float]) -> float:
    """Dot product via the C `dot`. Raise ValueError if the lengths differ."""
    # TODO: validate lengths, marshal both lists, call _lib().dot(...)
    raise NotImplementedError
