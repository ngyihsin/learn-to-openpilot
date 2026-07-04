"""Day 17 reference solution — ctypes wrappers around native.c."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile

_LIB = None


def _lib() -> ctypes.CDLL:
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
    n = len(values)
    arr = (ctypes.c_double * n)(*values)
    return _lib().array_sum(arr, n)


def py_scale(values: list[float], k: float) -> list[float]:
    n = len(values)
    arr = (ctypes.c_double * n)(*values)
    _lib().scale(arr, n, k)
    return list(arr)


def py_dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    ca = (ctypes.c_double * n)(*a)
    cb = (ctypes.c_double * n)(*b)
    return _lib().dot(ca, cb, n)
