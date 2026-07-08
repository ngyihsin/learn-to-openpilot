"""Day 36 reference solution — scaled dot-product self-attention.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.

Single-head, no batch dim: sequences are 2-D arrays of shape (seq_len, d).
"""
from __future__ import annotations

import numpy as np


def softmax(z, axis: int = -1):
    """Row-wise (or any-axis) numerically stable softmax."""
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """Return (output, weights).

        scores  = Q @ K.T / sqrt(d_k)     # how much each query matches each key
        weights = softmax(scores)         # rows sum to 1 (attention distribution)
        output  = weights @ V             # weighted blend of the values
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


def self_attention(X, Wq, Wk, Wv):
    """Self-attention: derive Q, K, V from the SAME input X via learned projections,
    then attend. Return (output, weights)."""
    X = np.asarray(X, dtype=float)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    return scaled_dot_product_attention(Q, K, V)
