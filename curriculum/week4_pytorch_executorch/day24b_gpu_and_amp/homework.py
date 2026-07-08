"""Day 24b homework — GPU & Mixed Precision (AMP).

Real models don't train on the CPU. The two habits that move you onto the GPU are:
(1) pick a ``torch.device`` once and move the model **and** every batch onto it, and
(2) run the forward pass under ``torch.autocast`` so the fast half-precision path is on
by default. Today you'll build both, plus the small helpers (parameter counting, a nested
``move_to``) you'll reach for every day after this.

Everything here runs correctly on CPU — the grader needs no GPU. On a real CUDA GPU the same
functions light up your RTX 5060; the only extra piece you'd add there (a ``GradScaler`` around
the backward pass) is described in the README.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

import torch


def pick_device(prefer_cuda: bool = True) -> torch.device:
    """Return the device to run on.

    Return ``torch.device('cuda')`` when ``prefer_cuda`` is true *and*
    ``torch.cuda.is_available()`` is true; otherwise return ``torch.device('cpu')``.
    This is the ONE place that mentions ``'cuda'`` — every other function takes whatever
    device you hand it, so the same code runs on CPU and GPU unchanged.
    """
    # TODO:
    #   - if prefer_cuda and torch.cuda.is_available(): return torch.device('cuda')
    #   - otherwise return torch.device('cpu')
    raise NotImplementedError


def move_to(obj, device):
    """Recursively move every tensor inside ``obj`` onto ``device``; return the result.

    A real batch is rarely a bare tensor — it's often a ``dict`` of tensors, or a
    ``list``/``tuple`` mixing tensors with metadata. Handle all of them:

        - a ``torch.Tensor``            -> ``obj.to(device)``
        - a ``list`` / ``tuple``        -> move each element, rebuild the SAME container type
        - a ``dict``                    -> move each value, keep the keys
        - anything else (int, str, ...) -> leave it exactly as-is

    Preserve the container type (a tuple stays a tuple, a list stays a list).
    """
    # TODO:
    #   - if obj is a torch.Tensor: return obj.to(device)
    #   - if obj is a list/tuple: recurse into each element and rebuild the same type
    #   - if obj is a dict: recurse into each value, keep the keys
    #   - otherwise: return obj unchanged
    raise NotImplementedError


def count_parameters(model, trainable_only: bool = True) -> int:
    """Return the total number of scalar parameters in ``model``.

    Sum ``p.numel()`` over ``model.parameters()``. When ``trainable_only`` is true, count only
    the parameters with ``requires_grad`` set (the ones an optimizer will actually update) — this
    is how you see the effect of freezing a backbone.
    """
    # TODO:
    #   - iterate over model.parameters()
    #   - if trainable_only, skip params whose requires_grad is False
    #   - return the sum of p.numel()
    raise NotImplementedError


def autocast_forward(model, x, device):
    """Run ``model(x)`` under mixed precision, without building a graph, and return the output.

    Wrap the call in BOTH ``torch.autocast(device_type=device.type)`` (so eligible ops run in
    half precision) and ``torch.no_grad()`` (this is inference — no gradients needed). Passing
    ``device.type`` keeps it device-agnostic: ``'cpu'`` on your laptop, ``'cuda'`` on the 5060.
    """
    # TODO:
    #   - with torch.autocast(device_type=device.type), torch.no_grad():
    #       return model(x)
    raise NotImplementedError


def amp_train_step(model, x, y, optimizer, loss_fn, device) -> float:
    """One AMP training step. Return the scalar loss as a Python float.

    Steps:
        1. forward under ``torch.autocast(device_type=device.type)``: ``pred = model(x)``
        2. ``loss = loss_fn(pred, y)``   (compute the loss in the autocast context too)
        3. ``optimizer.zero_grad()``
        4. ``loss.backward()``           (plain backward — see the note below)
        5. ``optimizer.step()``
        6. return ``loss.item()``

    NOTE: On a real CUDA GPU you'd wrap the backward pass in a ``torch.cuda.amp.GradScaler`` to
    stop half-precision gradients underflowing. This lesson runs on CPU, where scaling isn't
    needed, so keep the plain ``loss.backward()`` / ``optimizer.step()`` here. (See the README.)
    """
    # TODO:
    #   - with torch.autocast(device_type=device.type): pred = model(x); loss = loss_fn(pred, y)
    #   - optimizer.zero_grad(); loss.backward(); optimizer.step()
    #   - return loss.item()
    raise NotImplementedError
