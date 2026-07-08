"""Auto-grader for Day 25e — Building a Transformer Block.

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
scaled_dot_product_attention = _impl.scaled_dot_product_attention
TransformerEncoderBlock = _impl.TransformerEncoderBlock


def test_attention_matches_pytorch_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)

    out, _ = scaled_dot_product_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 4)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_weights_are_a_distribution():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)

    _, attn = scaled_dot_product_attention(q, k, v)
    assert attn.shape == (2, 3, 3)
    assert torch.all(attn >= 0)                       # softmax outputs are non-negative
    row_sums = attn.sum(dim=-1)                        # each query attends over all keys
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_block_output_shape():
    torch.manual_seed(0)
    block = TransformerEncoderBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
    block.eval()
    x = torch.randn(2, 5, 16)
    out = block(x)
    assert out.shape == (2, 5, 16)


def test_block_is_trainable():
    torch.manual_seed(0)
    block = TransformerEncoderBlock(d_model=16, n_heads=4, d_ff=32, dropout=0.0)
    block.train()
    x = torch.randn(2, 5, 16, requires_grad=True)

    out = block(x)
    assert out.requires_grad                          # the block is on the autograd graph
    out.sum().backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(grads) > 0                             # backward populated parameter gradients
