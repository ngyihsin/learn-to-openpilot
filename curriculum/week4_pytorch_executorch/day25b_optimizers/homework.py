"""Day 25b homework — Optimizers & Learning-Rate Scheduling.

The optimizer and its learning-rate schedule are the training-recipe knobs you tune most when
reproducing a paper. Today you'll (1) build the three optimizers you'll meet everywhere —
``SGD``, ``Adam``, ``AdamW`` — with weight decay, (2) compute two classic learning-rate schedules
(step decay and linear warmup) by hand, and (3) drive a real ``torch.optim.lr_scheduler`` and
record the learning rate it produces at each step.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

import torch


def make_optimizer(
    model: torch.nn.Module,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
):
    """Build a ``torch.optim`` optimizer over ``model.parameters()``.

    ``name`` is case-insensitive and one of ``"sgd"``, ``"adam"``, ``"adamw"``. Anything else
    should ``raise ValueError`` — an unknown optimizer name must fail loudly, not silently pick a
    default. Pass ``lr`` and ``weight_decay`` straight through to the optimizer constructor.
    """
    # TODO:
    #   - normalize name with name.lower()
    #   - "sgd"   -> torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    #   - "adam"  -> torch.optim.Adam(...)
    #   - "adamw" -> torch.optim.AdamW(...)
    #   - otherwise raise ValueError(f"unknown optimizer: {name!r}")
    raise NotImplementedError


def step_decay_lr(base_lr: float, step: int, drop: float = 0.5, every: int = 10) -> float:
    """Step-decay learning rate.

    The lr holds flat for ``every`` steps, then drops by a factor of ``drop``, and so on:

        base_lr * (drop ** (step // every))

    e.g. base 0.1, drop 0.5, every 10 -> 0.1 for steps 0..9, 0.05 for steps 10..19, ...
    Integer division ``//`` is the whole trick.
    """
    # TODO: return base_lr * (drop ** (step // every))
    raise NotImplementedError


def warmup_lr(base_lr: float, step: int, warmup_steps: int) -> float:
    """Linear-warmup learning rate.

    Ramp linearly from a small value up to ``base_lr`` over ``warmup_steps``, then hold:

        base_lr * min(1.0, (step + 1) / warmup_steps)

    The ``+1`` means step 0 isn't a dead zero. Warmup tames the large, noisy gradients you get at
    the very start of training when the weights are still random.
    """
    # TODO: return base_lr * min(1.0, (step + 1) / warmup_steps)
    raise NotImplementedError


def lr_schedule(optimizer, scheduler, n_steps: int) -> list:
    """Drive ``scheduler`` for ``n_steps`` and record the learning rate each step.

    A ``torch.optim.lr_scheduler`` mutates ``optimizer.param_groups[0]['lr']`` in place when you
    call ``scheduler.step()``. Loop ``n_steps`` times, and *each* iteration:
        1. read and record ``optimizer.param_groups[0]['lr']`` (the lr *before* stepping)
        2. call ``scheduler.step()``
    Return the list of recorded learning rates (length ``n_steps``).

    (PyTorch may print a warning that ``scheduler.step()`` was called before
    ``optimizer.step()``. Here we only *read* the schedule, so it's harmless — the recorded
    values are still correct. In real training you'd call ``optimizer.step()`` first.)
    """
    # TODO:
    #   - build an empty list
    #   - loop n_steps times: append optimizer.param_groups[0]['lr'], then scheduler.step()
    #   - return the list
    raise NotImplementedError
