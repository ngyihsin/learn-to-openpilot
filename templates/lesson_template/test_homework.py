"""Auto-grader for Day NN — <topic>.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

This loader pattern lets the whole 30-day suite run without module-name collisions and lets
``solution.py`` import sibling modules. Copy it verbatim into new lessons.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# import pytest  # uncomment when you need pytest.raises / approx / importorskip


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
solve = _impl.solve


def test_placeholder():
    # Replace with real behavior/edge-case tests. Make failure messages read like hints.
    assert solve is not None
