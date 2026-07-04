"""Auto-grader for Day 16 — C for Python programmers.

Compiles your C (``hot.c``) into a shared library and calls it via ctypes, comparing to a
Python reference. Grade the reference C with:  ``LP_IMPL=solution pytest -q``

Skips automatically if no C compiler is installed.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile

import pytest

_CC = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
if _CC is None:
    pytest.skip("no C compiler (gcc/cc/clang) available", allow_module_level=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "solution.c" if os.environ.get("LP_IMPL") == "solution" else "hot.c")


@pytest.fixture(scope="module")
def lib():
    tmp = tempfile.mkdtemp()
    so = os.path.join(tmp, "hot.so")
    subprocess.run([_CC, "-shared", "-fPIC", "-O2", _SRC, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.sum_squares.restype = ctypes.c_longlong
    lib.sum_squares.argtypes = [ctypes.c_long]
    lib.is_prime.restype = ctypes.c_int
    lib.is_prime.argtypes = [ctypes.c_long]
    lib.fib.restype = ctypes.c_longlong
    lib.fib.argtypes = [ctypes.c_int]
    return lib


def py_sum_squares(n: int) -> int:
    return sum(i * i for i in range(n))


def py_is_prime(n: int) -> int:
    if n < 2:
        return 0
    i = 2
    while i * i <= n:
        if n % i == 0:
            return 0
        i += 1
    return 1


def py_fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@pytest.mark.parametrize("n", [0, 1, 4, 10, 100, 1000, 100000])
def test_sum_squares(lib, n):
    assert lib.sum_squares(n) == py_sum_squares(n)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 17, 18, 25, 7919, 7920])
def test_is_prime(lib, n):
    assert lib.is_prime(n) == py_is_prime(n)


@pytest.mark.parametrize("n", [0, 1, 2, 10, 30, 50, 90])
def test_fib(lib, n):
    assert lib.fib(n) == py_fib(n)
