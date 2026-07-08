"""Auto-grader for Day 00 — the Python this course is written in.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


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
Backpack = _impl.Backpack
safe_divide = _impl.safe_divide


def test_init_and_add():
    b = Backpack(3)
    b.add("rope")
    b.add("torch")
    assert b.items == ["rope", "torch"]
    assert b.capacity == 3


def test_add_raises_when_full():
    b = Backpack(1)
    b.add("rope")
    with pytest.raises(ValueError):
        b.add("one too many")


def test_len_dunder():
    b = Backpack(5)
    assert len(b) == 0          # len() works because of __len__
    b.add("a")
    b.add("b")
    assert len(b) == 2


def test_getitem_dunder_and_index_error():
    b = Backpack(3)
    b.add("rope")
    assert b[0] == "rope"       # b[0] works because of __getitem__
    with pytest.raises(IndexError):
        b[1]                    # nothing there
    with pytest.raises(IndexError):
        b[-1]                   # keep it simple: negative indices are out of range here


def test_is_full_property():
    b = Backpack(2)
    assert b.is_full is False   # read WITHOUT parentheses — it's a property
    b.add("a")
    b.add("b")
    assert b.is_full is True


def test_safe_divide():
    assert safe_divide(6, 3) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        safe_divide(1, 0)
