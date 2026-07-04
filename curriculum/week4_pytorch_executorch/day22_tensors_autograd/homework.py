"""Day 22 homework — Tensors & Autograd.

PyTorch's superpower is **autograd**: you write the forward computation, and it computes the
gradients for you. Today you'll prove to yourself that autograd is *just calculus* by
(1) computing a gradient the slow numerical way and checking it matches, and
(2) training a line with a hand-written gradient-descent loop.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

from typing import Callable

import torch


def numerical_gradient(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Estimate ∇f(x) with the central-difference formula, element by element:

        grad[i] ≈ ( f(x + eps·e_i) − f(x − eps·e_i) ) / (2·eps)

    ``f`` maps a tensor to a scalar tensor. Return a tensor shaped like ``x``.
    This is the "ground truth" you check autograd against — real code uses autograd,
    but gradient-checking like this is how you catch bugs in custom layers.
    """
    # TODO:
    #   - make a float copy of x to perturb
    #   - for each index, bump it up by eps and down by eps, evaluate f, apply the formula
    #   - return the assembled gradient tensor
    raise NotImplementedError


def fit_line(
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.05,
    epochs: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``y ≈ w·x + b`` by gradient descent, using autograd for the gradients.

    Return the learned ``(w, b)`` as detached scalar tensors.

    The training-loop skeleton every PyTorch model shares:
        for each epoch:
            pred = w * x + b                 # forward
            loss = mean((pred - y) ** 2)     # mean squared error
            loss.backward()                  # autograd fills w.grad, b.grad
            with torch.no_grad():            # update weights without tracking it
                w -= lr * w.grad
                b -= lr * b.grad
            w.grad.zero_(); b.grad.zero_()   # gradients accumulate — clear them!
    """
    # TODO:
    #   - create scalar tensors w, b with requires_grad=True
    #   - run the loop above for `epochs` iterations
    #   - return w.detach(), b.detach()
    raise NotImplementedError
