"""Auto-grader for Day 25c — Saving & Loading Checkpoints.

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
save_checkpoint = _impl.save_checkpoint
load_checkpoint = _impl.load_checkpoint
models_allclose = _impl.models_allclose


def _make_mlp() -> torch.nn.Module:
    """A tiny 4→8→3 MLP. Its params are freshly (randomly) initialized on each call."""
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    )


def _train_a_few_steps(model, optimizer, seed=0):
    """Run a couple of deterministic SGD steps so the weights (and optimizer state) move."""
    torch.manual_seed(seed)
    x = torch.randn(16, 4)
    y = torch.randn(16, 3)
    loss_fn = torch.nn.MSELoss()
    for _ in range(3):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()


def test_fresh_models_differ():
    # Sanity: two independently initialized models are NOT allclose.
    torch.manual_seed(1)
    m1 = _make_mlp()
    torch.manual_seed(2)
    m2 = _make_mlp()
    assert models_allclose(m1, m2) is False


def test_models_allclose_true_for_same_weights():
    torch.manual_seed(0)
    m1 = _make_mlp()
    m2 = _make_mlp()
    m2.load_state_dict(m1.state_dict())
    assert models_allclose(m1, m2) is True


def test_models_allclose_key_mismatch_returns_false():
    torch.manual_seed(0)
    m1 = _make_mlp()
    m2 = torch.nn.Sequential(torch.nn.Linear(4, 8))  # different architecture / key set
    assert models_allclose(m1, m2) is False


def test_load_restores_weights_and_returns_epoch(tmp_path):
    torch.manual_seed(1)
    m1 = _make_mlp()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.037, momentum=0.9)
    _train_a_few_steps(m1, opt1)

    # A second fresh model with a different init must start out different.
    torch.manual_seed(2)
    m2 = _make_mlp()
    assert models_allclose(m1, m2) is False

    path = str(tmp_path / "ckpt.pt")
    assert save_checkpoint(path, m1, opt1, epoch=7) is None
    assert os.path.exists(path)

    # Fresh model + fresh optimizer, then load the checkpoint into them.
    torch.manual_seed(3)
    m3 = _make_mlp()
    opt3 = torch.optim.SGD(m3.parameters(), lr=0.001)  # deliberately different lr
    assert models_allclose(m1, m3) is False

    epoch = load_checkpoint(path, m3, opt3)
    assert epoch == 7
    assert models_allclose(m1, m3) is True

    # Optimizer hyperparameters came back too.
    assert opt3.param_groups[0]["lr"] == pytest.approx(opt1.param_groups[0]["lr"])


def test_reloaded_model_produces_identical_outputs(tmp_path):
    torch.manual_seed(4)
    m1 = _make_mlp()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.05)
    _train_a_few_steps(m1, opt1)

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, m1, opt1, epoch=12)

    torch.manual_seed(99)
    m3 = _make_mlp()
    opt3 = torch.optim.SGD(m3.parameters(), lr=0.05)
    assert load_checkpoint(path, m3, opt3) == 12

    torch.manual_seed(123)
    probe = torch.randn(5, 4)
    with torch.no_grad():
        out1 = m1(probe)
        out3 = m3(probe)
    assert torch.allclose(out1, out3)
