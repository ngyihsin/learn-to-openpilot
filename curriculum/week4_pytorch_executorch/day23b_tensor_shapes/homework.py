"""Day 23b homework — Tensor Shapes & Broadcasting.

The #1 thing that stops beginners cold in a real repo (YOLO, SAM, a Hugging Face model) is a
shape error: ``RuntimeError: The size of tensor a (3) must match the size of tensor b (4)``.
Today you build the two skills that make those errors trivial: reading a **shape** and
predicting a **broadcast**.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

import torch


def broadcast_shapes(a: tuple, b: tuple) -> tuple:
    """Compute the broadcasted result shape of two shapes ``a`` and ``b``.

    Implement the PyTorch / numpy rule yourself — do **not** call
    ``torch.broadcast_shapes``. That rule is the whole point of the exercise:

      1. Align the two shapes **from the right** (pad the shorter one with ``1``s on the left).
      2. For each aligned dimension, the two sizes are compatible if they're equal, or either
         one is ``1``; the output takes the larger of the two.
      3. If any aligned pair is incompatible, raise ``ValueError``.

    Examples: ``(3, 1)`` & ``(1, 4)`` -> ``(3, 4)``;  ``(2, 1, 4)`` & ``(5, 4)`` -> ``(2, 5, 4)``.
    """
    # TODO:
    #   - pad the shorter shape with 1s on the LEFT so both have the same length
    #   - walk the dimensions; for each pair, check compatibility (equal, or one is 1)
    #   - collect max(dim_a, dim_b); raise ValueError on any incompatible pair
    raise NotImplementedError


def batched_linear(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply one fully-connected layer to a batch: ``(N, in) -> (N, out)``.

    Shapes (PyTorch's ``nn.Linear`` convention):
        x : (N, in)     — a batch of N input vectors
        W : (out, in)   — the weight matrix, stored out-by-in
        b : (out,)      — the bias vector

    Return ``x @ W.T + b``. The bias ``(out,)`` broadcasts across all N rows — no loop needed.
    """
    # TODO:
    #   - transpose W to (in, out), matmul with x, then add the bias (it broadcasts)
    raise NotImplementedError


def to_shape(x: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Reshape ``x`` to ``shape``.

    A reshape is only valid when the total number of elements is unchanged. If
    ``x.numel()`` doesn't equal the product of ``shape``, raise ``ValueError`` with a message
    that reports **both** counts (so the error tells you what went wrong instead of making you
    guess). Otherwise return ``x.reshape(shape)``.
    """
    # TODO:
    #   - compute the target size as the product of `shape`
    #   - if it != x.numel(), raise ValueError mentioning both numbers
    #   - otherwise return x.reshape(shape)
    raise NotImplementedError


def channel_normalize(img: torch.Tensor) -> torch.Tensor:
    """Per-channel normalize an image ``(C, H, W)`` -> same shape.

    Subtract each channel's mean and divide by each channel's std — the exact preprocessing
    step in front of every vision model. Use ``keepdim`` broadcasting: reduce over the H and W
    axes (``dim=(1, 2), keepdim=True``) so the stats are shaped ``(C, 1, 1)`` and broadcast back
    over ``(C, H, W)``. Add ``eps = 1e-8`` to the std before dividing to avoid divide-by-zero.
    """
    # TODO:
    #   - compute per-channel mean and std with dim=(1, 2), keepdim=True
    #   - return (img - mean) / (std + 1e-8)
    raise NotImplementedError
