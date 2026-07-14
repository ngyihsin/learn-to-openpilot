"""Auto-grader for Day 28c — steering-angle regression.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
Skips if torch isn't installed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, random_split  # noqa: E402


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


def make_loaders():
    ds = _impl.make_road_dataset(512, seed=0)
    g = torch.Generator().manual_seed(1)
    train_ds, val_ds = random_split(ds, [448, 64], generator=g)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(2)),
        DataLoader(val_ds, batch_size=64, shuffle=False),
    )


def test_regressor_outputs_one_value_per_image():
    torch.manual_seed(0)
    model = _impl.build_regressor()
    out = model(torch.randn(4, 1, 16, 16))
    assert out.squeeze(-1).shape == (4,)   # one continuous value per image


def test_training_reaches_low_val_mae():
    torch.manual_seed(0)
    model = _impl.build_regressor()
    train_loader, val_loader = make_loaders()
    best_state, best_mae = _impl.train_regressor(model, train_loader, val_loader)
    # Predicting the mean label scores ~0.37 MAE; a working regressor gets far below.
    assert best_mae < 0.15, f"val MAE {best_mae:.3f} — did the loss/shapes go wrong?"


def test_best_state_reloads_to_reported_score():
    torch.manual_seed(0)
    model = _impl.build_regressor()
    train_loader, val_loader = make_loaders()
    best_state, best_mae = _impl.train_regressor(model, train_loader, val_loader)
    fresh = _impl.build_regressor()
    fresh.load_state_dict(best_state)
    assert abs(_impl.mae(fresh, val_loader) - best_mae) < 1e-4


def test_best_state_is_a_real_copy():
    torch.manual_seed(0)
    model = _impl.build_regressor()
    train_loader, val_loader = make_loaders()
    best_state, _ = _impl.train_regressor(model, train_loader, val_loader, epochs=2)
    live = model.state_dict()
    # A deepcopy/clone owns its own storage; a plain reference would alias the live weights.
    assert all(best_state[k].data_ptr() != live[k].data_ptr() for k in live)


def test_latency_is_a_real_measurement():
    torch.manual_seed(0)
    model = _impl.build_regressor()
    x = torch.randn(1, 1, 16, 16)
    lat = _impl.measure_latency_ms(model, x, warmup=2, iters=11)
    assert isinstance(lat, float)
    assert 0.0 < lat < 10_000.0            # a real, sane per-frame number


def test_frame_budget_arithmetic():
    assert _impl.meets_frame_budget(10.0, fps=20) is True    # 10 ms vs 50 ms budget
    assert _impl.meets_frame_budget(60.0, fps=20) is False   # blows the 20 Hz deadline
    assert _impl.meets_frame_budget(50.0, fps=20) is True    # exactly on budget still ships
    assert _impl.meets_frame_budget(30.0, fps=40) is False   # faster camera, tighter budget
