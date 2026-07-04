"""Auto-grader for Day 09 — process lifecycle state machine.

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
ProcessStateMachine = _impl.ProcessStateMachine
InvalidTransition = _impl.InvalidTransition


def test_happy_path_lifecycle():
    p = ProcessStateMachine("P1")
    assert p.run(["admit", "dispatch", "timeout", "dispatch", "exit"]) == "terminated"
    assert p.is_terminated()
    assert p.history == ["new", "ready", "running", "ready", "running", "terminated"]


def test_io_wait_cycle():
    p = ProcessStateMachine("P2")
    p.run(["admit", "dispatch", "wait", "wakeup", "dispatch", "exit"])
    assert p.state == "terminated"


def test_context_switch_count():
    p = ProcessStateMachine("P3")
    # dispatched three separate times -> 3 CPU grants.
    p.run(["admit", "dispatch", "timeout", "dispatch", "wait", "wakeup", "dispatch", "exit"])
    assert p.context_switches == 3


def test_can_reports_legality():
    p = ProcessStateMachine("P4")
    assert p.can("admit")
    assert not p.can("dispatch")     # can't dispatch from 'new'
    p.on("admit")
    assert p.can("dispatch")


@pytest.mark.parametrize("bad_first_event", ["dispatch", "timeout", "wait", "wakeup", "exit"])
def test_illegal_from_new_raises(bad_first_event):
    p = ProcessStateMachine("P5")
    with pytest.raises(InvalidTransition):
        p.on(bad_first_event)
    assert p.state == "new"          # state unchanged after a rejected transition


def test_cannot_run_terminated_process():
    p = ProcessStateMachine("P6")
    p.run(["admit", "dispatch", "exit"])
    for event in ("admit", "dispatch", "wait"):
        with pytest.raises(InvalidTransition):
            p.on(event)


def test_waiting_cannot_be_dispatched_directly():
    # A blocked process must be woken (-> ready) before it can run again.
    p = ProcessStateMachine("P7")
    p.run(["admit", "dispatch", "wait"])
    assert p.state == "waiting"
    with pytest.raises(InvalidTransition):
        p.on("dispatch")
