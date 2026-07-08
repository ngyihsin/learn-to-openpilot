"""Auto-grader for Day 25d — Transfer Learning & Fine-Tuning.

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
freeze = _impl.freeze
unfreeze = _impl.unfreeze
replace_head = _impl.replace_head
trainable_param_names = _impl.trainable_param_names
finetune_step = _impl.finetune_step


class TinyNet(torch.nn.Module):
    """A stand-in for a real pretrained model: a backbone plus a swappable ``.head``."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 8)
        self.head = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


def test_freeze_then_replace_head_leaves_only_head_trainable():
    torch.manual_seed(0)
    model = TinyNet()
    freeze(model.backbone)
    replace_head(model, torch.nn.Linear(8, 3))
    assert trainable_param_names(model) == ["head.bias", "head.weight"]


def test_replace_head_changes_output_shape():
    torch.manual_seed(0)
    model = TinyNet()
    replace_head(model, torch.nn.Linear(8, 3))
    out = model(torch.randn(5, 4))
    assert out.shape == (5, 3)


def test_finetune_moves_head_but_not_frozen_backbone():
    torch.manual_seed(0)
    model = TinyNet()
    freeze(model.backbone)
    replace_head(model, torch.nn.Linear(8, 3))

    backbone_before = model.backbone.weight.detach().clone()
    head_before = model.head.weight.detach().clone()

    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss = finetune_step(model, x, y, optimizer, loss_fn)
    assert isinstance(loss, float)

    # Frozen backbone must not move; the fresh head must have learned.
    assert torch.allclose(model.backbone.weight, backbone_before)
    assert not torch.allclose(model.head.weight, head_before)


def test_finetune_step_returns_the_loss_value():
    torch.manual_seed(1)
    model = TinyNet()
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

    expected = loss_fn(model(x), y).item()
    got = finetune_step(model, x, y, optimizer, loss_fn)
    assert got == pytest.approx(expected, rel=1e-5)


def test_unfreeze_makes_backbone_trainable_again():
    torch.manual_seed(0)
    model = TinyNet()
    freeze(model.backbone)
    assert trainable_param_names(model) == ["head.bias", "head.weight"]

    unfreeze(model.backbone)
    assert trainable_param_names(model) == [
        "backbone.bias",
        "backbone.weight",
        "head.bias",
        "head.weight",
    ]
