"""Day 25e homework — Building a Transformer Block.

The transformer block is the single most reused module in modern vision-language models:
CLIP, ViT, and every VLM the lab works on are just stacks of it. Today you'll build one
from the ground up — first the **scaled-dot-product attention** kernel by hand (the same math
you derived on Day 36), then wrap PyTorch's multi-head attention into a real, reusable
**encoder block**: attention → residual → norm → feed-forward → residual → norm.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled-dot-product attention from scratch.

    ``q``, ``k``, ``v`` each have shape ``(B, T, d)`` — a batch of ``B`` sequences, ``T``
    tokens each, every token a ``d``-dimensional vector. The recipe (Day 36's math, in code):

        scores = q @ k.transpose(-2, -1) / sqrt(d)   # (B, T, T): token i vs token j
        attn   = softmax(scores, dim=-1)             # each row is a probability distribution
        output = attn @ v                            # (B, T, d): weighted mix of values

    Return the tuple ``(output, attn_weights)`` where ``output`` is ``(B, T, d)`` and
    ``attn_weights`` is ``(B, T, T)``. Implement it yourself — do **not** call
    ``torch.nn.functional.scaled_dot_product_attention``.
    """
    # TODO:
    #   - d = q.size(-1); compute raw scores with q @ k.transpose(-2, -1)
    #   - divide by sqrt(d) so the softmax doesn't saturate as d grows
    #   - softmax over the LAST dim (dim=-1) to get attn_weights
    #   - multiply attn_weights @ v to get the output
    #   - return (output, attn_weights)
    raise NotImplementedError


class TransformerEncoderBlock(nn.Module):
    """One post-norm transformer encoder block — the unit every VLM stacks.

    ``__init__(self, d_model, n_heads, d_ff, dropout=0.0)`` should build:
      - ``nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)``
        for the self-attention sublayer,
      - a position-wise feed-forward network: ``Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)``,
      - two ``nn.LayerNorm(d_model)`` layers (one per sublayer).

    ``forward(x)`` takes ``x`` of shape ``(B, T, d_model)`` and applies the **post-norm**
    sandwich (normalize *after* adding the residual):

        x = LayerNorm( x + SelfAttn(x) )        # attention sublayer + residual
        x = LayerNorm( x + FFN(x) )             # feed-forward sublayer + residual

    Return the result, shape ``(B, T, d_model)``. The residual (``x + ...``) is what lets you
    stack dozens of these without the gradient vanishing.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        # TODO:
        #   - self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        #   - self.ff   = nn.Sequential(Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model))
        #   - self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        #   - self-attention: attn_out, _ = self.attn(x, x, x)   (query=key=value=x)
        #   - x = self.norm1(x + attn_out)
        #   - x = self.norm2(x + self.ff(x))
        #   - return x
        raise NotImplementedError
