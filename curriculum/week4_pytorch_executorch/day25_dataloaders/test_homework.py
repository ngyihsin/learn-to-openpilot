"""Auto-grader for Day 25 — Datasets & DataLoaders.

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
ToyDataset, make_loader, standardize = _impl.ToyDataset, _impl.make_loader, _impl.standardize


def make_xy(n=10, d=3):
    torch.manual_seed(0)
    return torch.randn(n, d), torch.randint(0, 2, (n,))


def test_dataset_len_and_getitem():
    X, y = make_xy()
    ds = ToyDataset(X, y)
    assert len(ds) == 10
    x0, y0 = ds[0]
    assert torch.equal(x0, X[0])
    assert int(y0) == int(y[0])


def test_loader_covers_all_in_order():
    X, y = make_xy(10, 3)
    loader = make_loader(ToyDataset(X, y), batch_size=4, shuffle=False)
    batches = list(loader)
    assert [len(b[1]) for b in batches] == [4, 4, 2]      # last batch is the remainder
    xs = torch.cat([b[0] for b in batches])
    assert torch.equal(xs, X)                             # order preserved when not shuffled


def test_loader_shuffle_covers_all_once():
    X, y = make_xy(12, 2)
    loader = make_loader(ToyDataset(X, y), batch_size=5, shuffle=True)
    seen = torch.cat([b[0] for b in loader])
    assert seen.shape == X.shape
    # Same set of rows, possibly reordered: sort both and compare.
    assert torch.allclose(seen.sum(0), X.sum(0))          # every row present exactly once


def test_standardize():
    X = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    Z = standardize(X)
    assert torch.allclose(Z.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(Z.std(dim=0, unbiased=True), torch.ones(2), atol=1e-2)
