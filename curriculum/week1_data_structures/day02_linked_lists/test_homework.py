"""Auto-grader for Day 02 — Doubly-Linked List.

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


DoublyLinkedList = _load(os.environ.get("LP_IMPL", "homework")).DoublyLinkedList


def _assert_intact(dll, expected: list):
    """Forward and backward traversals must agree — proves prev/next stayed consistent."""
    assert dll.to_list() == expected
    assert dll.to_list_reverse() == expected[::-1]
    assert len(dll) == len(expected)


def test_empty():
    _assert_intact(DoublyLinkedList(), [])


def test_push_back_and_front():
    dll = DoublyLinkedList()
    for v in (2, 3, 4):
        dll.push_back(v)
    dll.push_front(1)
    _assert_intact(dll, [1, 2, 3, 4])


def test_pop_front_and_back():
    dll = DoublyLinkedList()
    for v in (1, 2, 3, 4):
        dll.push_back(v)
    assert dll.pop_front() == 1
    assert dll.pop_back() == 4
    _assert_intact(dll, [2, 3])


def test_pop_until_empty_keeps_pointers_sane():
    dll = DoublyLinkedList()
    dll.push_back(7)
    assert dll.pop_back() == 7
    _assert_intact(dll, [])
    # Reuse after emptying must still work (head/tail were reset to None).
    dll.push_front(9)
    _assert_intact(dll, [9])


@pytest.mark.parametrize("popper", ["pop_front", "pop_back"])
def test_pop_from_empty_raises(popper):
    dll = DoublyLinkedList()
    with pytest.raises(IndexError):
        getattr(dll, popper)()


def test_delete_value_head_middle_tail_and_missing():
    dll = DoublyLinkedList()
    for v in (1, 2, 3, 4, 5):
        dll.push_back(v)

    assert dll.delete_value(3) is True     # middle
    _assert_intact(dll, [1, 2, 4, 5])
    assert dll.delete_value(1) is True      # head
    _assert_intact(dll, [2, 4, 5])
    assert dll.delete_value(5) is True      # tail
    _assert_intact(dll, [2, 4])
    assert dll.delete_value(99) is False    # missing
    _assert_intact(dll, [2, 4])


def test_delete_only_removes_first_match():
    dll = DoublyLinkedList()
    for v in (1, 2, 1, 3):
        dll.push_back(v)
    assert dll.delete_value(1) is True
    _assert_intact(dll, [2, 1, 3])


def test_reverse():
    dll = DoublyLinkedList()
    for v in (1, 2, 3, 4):
        dll.push_back(v)
    dll.reverse()
    _assert_intact(dll, [4, 3, 2, 1])
    # After reversing, ends still behave — push/pop keep working.
    dll.push_back(0)
    assert dll.pop_front() == 4
    _assert_intact(dll, [3, 2, 1, 0])
