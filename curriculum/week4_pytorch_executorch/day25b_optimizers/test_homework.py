"""Auto-grader for Day 25b — Optimizers & Learning-Rate Scheduling.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

Skips automatically if torch isn't installed (it's only needed for Week 4).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")  # Week-4 only dependency


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
make_optimizer = _impl.make_optimizer
step_decay_lr = _impl.step_decay_lr
warmup_lr = _impl.warmup_lr
lr_schedule = _impl.lr_schedule


def _tiny_model():
    return torch.nn.Linear(2, 2)


def test_make_optimizer_types():
    m = _tiny_model()
    assert isinstance(make_optimizer(m, "sgd", 0.1), torch.optim.SGD)
    assert isinstance(make_optimizer(m, "adam", 0.1), torch.optim.Adam)
    assert isinstance(make_optimizer(m, "adamw", 0.1), torch.optim.AdamW)


def test_make_optimizer_case_insensitive():
    m = _tiny_model()
    assert isinstance(make_optimizer(m, "SGD", 0.1), torch.optim.SGD)
    assert isinstance(make_optimizer(m, "AdamW", 0.1), torch.optim.AdamW)


def test_make_optimizer_sets_lr_and_weight_decay():
    m = _tiny_model()
    opt = make_optimizer(m, "adam", lr=0.003, weight_decay=0.01)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.003)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_make_optimizer_bad_name_raises():
    m = _tiny_model()
    with pytest.raises(ValueError):
        make_optimizer(m, "rmsprop", 0.1)


def test_step_decay_lr_values():
    assert step_decay_lr(0.1, 0) == pytest.approx(0.1)
    assert step_decay_lr(0.1, 9) == pytest.approx(0.1)
    assert step_decay_lr(0.1, 10) == pytest.approx(0.05)
    assert step_decay_lr(0.1, 25) == pytest.approx(0.025)


def test_warmup_lr_values():
    assert warmup_lr(0.1, 0, 5) == pytest.approx(0.02)
    assert warmup_lr(0.1, 4, 5) == pytest.approx(0.1)
    assert warmup_lr(0.1, 10, 5) == pytest.approx(0.1)


def test_lr_schedule_step_lr():
    opt = torch.optim.SGD(_tiny_model().parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    lrs = lr_schedule(opt, sched, n_steps=6)
    assert len(lrs) == 6
    expected = [0.1, 0.1, 0.01, 0.01, 0.001, 0.001]
    for got, want in zip(lrs, expected):
        assert got == pytest.approx(want)
