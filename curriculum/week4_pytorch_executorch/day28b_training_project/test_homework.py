"""Auto-grader for Day 28b — A Full Training Project (End-to-End).

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
make_dataset = _impl.make_dataset
split_dataset = _impl.split_dataset
build_model = _impl.build_model
evaluate = _impl.evaluate
train = _impl.train

from torch.utils.data import DataLoader


def test_make_dataset_shapes_and_dtypes():
    ds = make_dataset(256, seed=0)
    assert len(ds) == 256
    X, y = ds.tensors
    assert X.shape == (256, 2)
    assert X.dtype == torch.float32
    assert y.shape == (256,)
    assert y.dtype == torch.long
    assert set(int(v) for v in torch.unique(y).tolist()) <= {0, 1}


def test_make_dataset_reproducible():
    a = make_dataset(128, seed=7).tensors[0]
    b = make_dataset(128, seed=7).tensors[0]
    assert torch.allclose(a, b)


def test_split_sizes_sum_and_partition():
    ds = make_dataset(256, seed=0)
    train_ds, val_ds = split_dataset(ds, val_frac=0.25, seed=0)
    assert len(train_ds) == 192
    assert len(val_ds) == 64
    assert len(train_ds) + len(val_ds) == len(ds)


def test_build_model_outputs_logits():
    torch.manual_seed(0)
    model = build_model()
    out = model(torch.randn(5, 2))
    assert out.shape == (5, 2)


def _run(seed: int = 0):
    torch.manual_seed(seed)
    ds = make_dataset(512, seed=seed)
    train_ds, val_ds = split_dataset(ds, val_frac=0.2, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = build_model()
    hist = train(model, train_loader, val_loader, epochs=15, lr=0.1)
    return hist, val_loader


def test_history_shape_and_keys():
    hist, _ = _run()
    for key in ("train_loss", "val_acc", "best_val_acc", "best_state"):
        assert key in hist
    assert len(hist["train_loss"]) == 15
    assert len(hist["val_acc"]) == 15
    assert hist["best_val_acc"] == pytest.approx(max(hist["val_acc"]))


def test_training_learns_and_loss_drops():
    hist, _ = _run()
    assert hist["best_val_acc"] > 0.9          # separable blobs -> easy
    assert hist["train_loss"][-1] < hist["train_loss"][0]  # loss went down


def test_best_state_restores_accuracy():
    hist, val_loader = _run()
    fresh = build_model()
    fresh.load_state_dict(hist["best_state"])
    assert evaluate(fresh, val_loader) >= 0.9
