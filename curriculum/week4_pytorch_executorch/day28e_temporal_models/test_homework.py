"""Auto-grader for Day 28e — temporal models.

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
    ds = _impl.make_motion_dataset(512, seed=0)
    g = torch.Generator().manual_seed(1)
    train_ds, val_ds = random_split(ds, [448, 64], generator=g)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(2)),
        DataLoader(val_ds, batch_size=64, shuffle=False),
    )


def test_model_shapes():
    torch.manual_seed(0)
    single = _impl.build_single_frame_regressor()
    stacked = _impl.build_stacked_regressor(_impl.T)
    rec = _impl.TemporalRegressor()
    clips = torch.randn(5, _impl.T, 16, 16)
    assert single(clips[:, -1:]).squeeze(-1).shape == (5,)
    assert stacked(clips).squeeze(-1).shape == (5,)
    assert rec(clips).squeeze(-1).shape == (5,)


def test_init_state_is_fresh_zeros():
    torch.manual_seed(0)
    rec = _impl.TemporalRegressor()
    h = rec.init_state(7)
    assert h.shape == (7, rec.d)
    assert torch.allclose(h, torch.zeros(7, rec.d))


def test_step_is_a_pure_explicit_state_function():
    torch.manual_seed(0)
    rec = _impl.TemporalRegressor().eval()
    frame = torch.randn(3, 16, 16)
    h0 = rec.init_state(3)
    with torch.no_grad():
        p1, h1 = rec.step(frame, h0)
        p2, h2 = rec.step(frame, h0)      # same inputs -> same outputs, no hidden mutation
    assert torch.allclose(p1, p2) and torch.allclose(h1, h2)
    assert p1.shape == (3, 1) and h1.shape == (3, rec.d)
    assert not torch.allclose(h1, h0)     # the state actually advances


def test_forward_equals_manually_threading_step():
    torch.manual_seed(0)
    rec = _impl.TemporalRegressor().eval()
    clips = torch.randn(4, _impl.T, 16, 16)
    with torch.no_grad():
        out = rec(clips)
        h = rec.init_state(4)
        manual = None
        for t in range(_impl.T):
            manual, h = rec.step(clips[:, t], h)
    assert torch.allclose(out, manual, atol=1e-6)


@pytest.mark.slow
def test_single_frame_model_is_blind_to_motion():
    """The proof: speed isn't in a single frame, so this model cannot beat ~guessing."""
    train_loader, val_loader = make_loaders()
    torch.manual_seed(3)
    single = _impl.build_single_frame_regressor()
    single_mae = _impl.train_regressor(single, train_loader, val_loader, last_frame_only=True)
    assert single_mae > 0.3, (
        f"single-frame val MAE {single_mae:.3f} — it should NOT be able to learn speed"
    )


@pytest.mark.slow
def test_frame_stacking_sees_motion():
    train_loader, val_loader = make_loaders()
    torch.manual_seed(3)
    stacked = _impl.build_stacked_regressor(_impl.T)
    stacked_mae = _impl.train_regressor(stacked, train_loader, val_loader)
    assert stacked_mae < 0.15, f"stacked val MAE {stacked_mae:.3f} — should learn speed easily"


@pytest.mark.slow
def test_recurrent_state_sees_motion():
    train_loader, val_loader = make_loaders()
    torch.manual_seed(3)
    rec = _impl.TemporalRegressor()
    rec_mae = _impl.train_regressor(rec, train_loader, val_loader)
    assert rec_mae < 0.15, f"recurrent val MAE {rec_mae:.3f} — should learn speed easily"
