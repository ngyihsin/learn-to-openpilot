"""Auto-grader for Day 24b — GPU & Mixed Precision (AMP).

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
pick_device = _impl.pick_device
move_to = _impl.move_to
count_parameters = _impl.count_parameters
autocast_forward = _impl.autocast_forward
amp_train_step = _impl.amp_train_step


import torch.nn as nn


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))


def test_pick_device_returns_valid_device():
    dev = pick_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cpu", "cuda"}


def test_pick_device_cpu_when_not_prefer_cuda():
    dev = pick_device(prefer_cuda=False)
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"


def test_move_to_nested_dict():
    cpu = torch.device("cpu")
    batch = {"x": torch.randn(4), "meta": [torch.ones(2), 5]}
    out = move_to(batch, cpu)

    assert out["x"].device.type == "cpu"
    assert out["meta"][0].device.type == "cpu"
    assert out["meta"][1] == 5            # the int is left untouched
    assert isinstance(out["meta"], list)  # container type preserved


def test_move_to_preserves_tuple_and_passthrough():
    cpu = torch.device("cpu")
    obj = (torch.zeros(3), "label", None)
    out = move_to(obj, cpu)
    assert isinstance(out, tuple)
    assert out[0].device.type == "cpu"
    assert out[1] == "label"
    assert out[2] is None


def test_count_parameters_linear():
    layer = nn.Linear(4, 3)  # weight 4*3 = 12, bias 3  ->  15
    assert count_parameters(layer) == 15
    assert count_parameters(layer, trainable_only=False) == 15


def test_count_parameters_respects_frozen():
    layer = nn.Linear(4, 3)
    layer.bias.requires_grad_(False)                 # freeze the 3 bias params
    assert count_parameters(layer, trainable_only=True) == 12
    assert count_parameters(layer, trainable_only=False) == 15


def test_autocast_forward_shape():
    dev = torch.device("cpu")
    model = _tiny_model()
    x = torch.randn(5, 3)
    out = autocast_forward(model, x, dev)
    assert out.shape == (5, 2)


def test_amp_train_step_updates_and_returns_finite_loss():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = nn.Linear(3, 2)
    x = torch.randn(16, 3)
    y = torch.randn(16, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    before = model.weight.detach().clone()
    loss = amp_train_step(model, x, y, optimizer, loss_fn, dev)

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.allclose(before, model.weight)  # at least one param moved
