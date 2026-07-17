"""Auto-grader for Day 28d — QAT from scratch.

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
import torch.nn as nn  # noqa: E402
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


def test_quantize_tensor_sits_on_the_grid():
    torch.manual_seed(0)
    x = torch.randn(64)
    for bits in (3, 8):
        q = _impl.quantize_tensor(x, bits)
        qmax = 2 ** (bits - 1) - 1
        scale = x.abs().max() / qmax
        # every value is (integer in [-qmax, qmax]) * scale
        ints = q / scale
        assert torch.allclose(ints, ints.round(), atol=1e-5)
        assert ints.abs().max() <= qmax + 1e-5
        # rounding error is at most half a grid step
        assert (q - x).abs().max() <= scale / 2 + 1e-6


def test_quantize_tensor_handles_zeros():
    q = _impl.quantize_tensor(torch.zeros(8), 8)
    assert torch.isfinite(q).all() and torch.allclose(q, torch.zeros(8))


def test_ste_lets_gradients_through():
    x = torch.randn(16, requires_grad=True)
    y = _impl.STEQuant.apply(x, 3)
    y.sum().backward()
    # true grad of round() is 0 a.e.; the STE must pass it straight through instead
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_qat_linear_quantizes_forward_but_keeps_float_storage():
    torch.manual_seed(0)
    layer = _impl.QATLinear(8, 4, bits=3)
    x = torch.randn(5, 8)
    out = layer(x)
    assert out.shape == (5, 4)
    # forward must use the quantized weight, not the raw one...
    w_q = _impl.quantize_tensor(layer.weight, 3)
    expected = nn.functional.linear(x, w_q, layer.bias)
    assert torch.allclose(out, expected, atol=1e-5)
    # ...while the stored (shadow) weight stays off-grid float
    assert not torch.allclose(layer.weight, w_q)
    # and gradients reach the shadow weight through the STE
    out.sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_ptq_does_not_mutate_the_original():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    before = [p.clone() for p in model.parameters()]
    q = _impl.ptq_quantize(model, bits=3)
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))
    # and the copy's weights really are on the grid
    for m, qm in zip(model.modules(), q.modules()):
        if isinstance(m, nn.Linear):
            assert torch.allclose(qm.weight, _impl.quantize_tensor(m.weight, 3), atol=1e-6)


@pytest.mark.slow
def test_qat_recovers_what_ptq_loses_at_3_bits():
    """The headline experiment: float vs PTQ vs QAT on the road regressor at 3 bits."""
    train_loader, val_loader = make_loaders()

    torch.manual_seed(3)
    float_model = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1))
    float_mae = _impl.train_regressor(float_model, train_loader, val_loader)
    ptq_mae = _impl.mae(_impl.ptq_quantize(float_model, bits=3), val_loader)

    torch.manual_seed(3)
    qat_model = _impl.build_qat_regressor(bits=3)
    qat_mae = _impl.train_regressor(qat_model, train_loader, val_loader)

    assert float_mae < 0.05                     # sanity: the task is learnable
    assert ptq_mae > 2 * float_mae              # 3-bit PTQ visibly hurts
    assert qat_mae < 0.5 * ptq_mae, (           # QAT recovers most of the damage
        f"float={float_mae:.4f} ptq={ptq_mae:.4f} qat={qat_mae:.4f}"
    )
