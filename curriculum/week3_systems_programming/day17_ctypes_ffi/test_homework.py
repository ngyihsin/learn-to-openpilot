"""Auto-grader for Day 17 — ctypes FFI.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

Skips automatically if no C compiler is installed.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys

import pytest

if not (shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")):
    pytest.skip("no C compiler (gcc/cc/clang) available", allow_module_level=True)


def _load(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load(os.environ.get("LP_IMPL", "homework"))
py_array_sum, py_scale, py_dot = _impl.py_array_sum, _impl.py_scale, _impl.py_dot


def test_array_sum():
    assert py_array_sum([1.0, 2.0, 3.0, 4.0]) == pytest.approx(10.0)
    assert py_array_sum([]) == pytest.approx(0.0)


def test_scale_returns_mutated_values():
    out = py_scale([1.0, 2.0, 3.0], 2.5)
    assert out == pytest.approx([2.5, 5.0, 7.5])


def test_dot():
    assert py_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)


def test_dot_length_mismatch():
    with pytest.raises(ValueError):
        py_dot([1.0, 2.0], [1.0])


def test_larger_arrays_match_python():
    xs = [float(i) for i in range(500)]
    ys = [float(i) * 0.5 for i in range(500)]
    assert py_array_sum(xs) == pytest.approx(sum(xs))
    assert py_dot(xs, ys) == pytest.approx(sum(a * b for a, b in zip(xs, ys)))
    assert py_scale(xs, 3.0) == pytest.approx([x * 3.0 for x in xs])
