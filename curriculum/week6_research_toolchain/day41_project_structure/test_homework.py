"""Auto-grader for Day 41 — reproducibility utilities.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys

import numpy as np
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
set_seed = _impl.set_seed
AverageMeter = _impl.AverageMeter
merge_config = _impl.merge_config
pick_best = _impl.pick_best


def _sample():
    return (random.random(), round(float(np.random.rand()), 12))


def test_set_seed_makes_runs_reproducible():
    set_seed(0)
    a = _sample()
    set_seed(0)
    b = _sample()
    assert a == b                      # same seed -> identical draws (python AND numpy)
    set_seed(1)
    c = _sample()
    assert a != c                      # a different seed differs


def test_average_meter():
    m = AverageMeter()
    m.update(2.0)
    m.update(4.0)
    assert m.avg == pytest.approx(3.0)
    m.update(6.0, n=2)                 # counts as two samples of 6
    assert m.avg == pytest.approx((2 + 4 + 12) / 4)
    m.reset()
    assert m.avg == 0.0                # empty meter is 0, not a crash


def test_merge_config_overrides_and_is_pure():
    defaults = {"lr": 0.1, "epochs": 10, "model": "resnet18"}
    overrides = {"lr": 0.01, "seed": 42}
    merged = merge_config(defaults, overrides)
    assert merged == {"lr": 0.01, "epochs": 10, "model": "resnet18", "seed": 42}
    # inputs must be untouched
    assert defaults == {"lr": 0.1, "epochs": 10, "model": "resnet18"}
    assert overrides == {"lr": 0.01, "seed": 42}


def test_pick_best():
    history = [
        {"epoch": 1, "val_loss": 0.50, "acc": 0.80},
        {"epoch": 2, "val_loss": 0.30, "acc": 0.91},
        {"epoch": 3, "val_loss": 0.40, "acc": 0.88},
    ]
    assert pick_best(history, "val_loss", mode="min")["epoch"] == 2
    assert pick_best(history, "acc", mode="max")["epoch"] == 2
    with pytest.raises(ValueError):
        pick_best([], "val_loss")
    with pytest.raises(ValueError):
        pick_best(history, "val_loss", mode="sideways")
