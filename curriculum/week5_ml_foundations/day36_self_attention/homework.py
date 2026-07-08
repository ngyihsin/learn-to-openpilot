"""Day 36 homework — scaled dot-product self-attention.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

Attention is the layer behind Transformers — and behind the models you'll run in Week 7 (SAM,
Grounding DINO are all Transformer-based). The idea: every position looks at every other position and
blends their information, weighted by how relevant each one is. You build it from three arrays —
**Q**ueries, **K**eys, **V**alues — with one matrix multiply, a softmax, and another matrix multiply.

Single-head, no batch dim: a sequence is a 2-D array of shape (seq_len, d).
"""
from __future__ import annotations

import numpy as np


def softmax(z, axis: int = -1):
    """Numerically stable softmax along ``axis`` (subtract the max along that axis first).
    Works on 2-D input: with axis=-1 each ROW becomes a probability distribution."""
    # TODO: subtract z.max(axis=axis, keepdims=True), exponentiate, divide by the sum along axis
    raise NotImplementedError


def scaled_dot_product_attention(Q, K, V):
    """Return (output, weights).

        d_k     = Q.shape[-1]
        scores  = Q @ K.T / sqrt(d_k)     # (n_queries, n_keys) — match each query to each key
        weights = softmax(scores, axis=-1)# each query's weights over the keys sum to 1
        output  = weights @ V             # (n_queries, d_v)   — blend the values
    """
    # TODO
    raise NotImplementedError


def self_attention(X, Wq, Wk, Wv):
    """Self-attention: build Q, K, V from the SAME input X by projecting it:
        Q = X @ Wq,  K = X @ Wk,  V = X @ Wv
    then run scaled_dot_product_attention. Return (output, weights)."""
    # TODO
    raise NotImplementedError
