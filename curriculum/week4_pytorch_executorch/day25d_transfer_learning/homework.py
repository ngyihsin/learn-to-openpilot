"""Day 25d homework — Transfer Learning & Fine-Tuning.

The most common workflow in real research isn't training from scratch — it's taking a model
someone else already trained (a YOLO backbone, a CLIP encoder), **freezing** most of it,
**replacing the head** with one shaped for your task, and **fine-tuning**. Today you'll build
the five primitives that make that possible, and prove that a frozen backbone stays frozen while
the new head learns.

These functions operate on any model that exposes a ``.head`` attribute (like the ``TinyNet`` the
grader defines). Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``.)
"""
from __future__ import annotations

import torch


def freeze(module: torch.nn.Module) -> None:
    """Freeze ``module``: set ``requires_grad = False`` on every one of its parameters.

    A frozen parameter gets no gradient from ``backward()``, so the optimizer can't move it.
    """
    # TODO:
    #   - loop over module.parameters() and set p.requires_grad = False
    raise NotImplementedError


def unfreeze(module: torch.nn.Module) -> None:
    """Unfreeze ``module``: set ``requires_grad = True`` on every one of its parameters.

    The inverse of ``freeze`` — use it when you want to fine-tune deeper layers too.
    """
    # TODO:
    #   - loop over module.parameters() and set p.requires_grad = True
    raise NotImplementedError


def replace_head(model: torch.nn.Module, new_head: torch.nn.Module) -> None:
    """Swap ``model.head`` for ``new_head`` (sized for your task's outputs).

    A freshly constructed layer has trainable parameters by default, so the new head will train.
    """
    # TODO:
    #   - assign new_head to model.head
    raise NotImplementedError


def trainable_param_names(model: torch.nn.Module) -> list:
    """Return a **sorted** list of the names of parameters that are currently trainable.

    Read from ``model.named_parameters()`` and keep the names where ``requires_grad`` is True.
    After freezing the backbone and swapping the head, this is how you confirm that *only* the
    head will learn.
    """
    # TODO:
    #   - collect name for each (name, p) in model.named_parameters() if p.requires_grad
    #   - return them sorted
    raise NotImplementedError


def finetune_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
) -> float:
    """Run one fine-tuning step and return the scalar loss value.

    The canonical step, identical to any training loop:
        optimizer.zero_grad()        # clear last step's gradients
        pred = model(x)              # forward
        loss = loss_fn(pred, y)      # measure error
        loss.backward()              # autograd fills .grad for trainable params only
        optimizer.step()             # update — frozen params have no grad, so they don't move

    Return ``loss.item()`` (a plain float).
    """
    # TODO:
    #   - zero_grad, forward, loss, backward, step
    #   - return loss.item()
    raise NotImplementedError
