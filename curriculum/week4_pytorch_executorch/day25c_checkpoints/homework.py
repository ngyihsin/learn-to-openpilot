"""Day 25c homework — Saving & Loading Checkpoints.

A checkpoint is how you survive a crash, resume a multi-hour run, or ship the exact weights from
your best epoch. PyTorch serializes a model through its **`state_dict`** — a name→tensor mapping of
every parameter and buffer. Today you'll bundle the model's `state_dict`, the optimizer's
`state_dict`, and the epoch into one file, then reload all of it into a fresh model and prove it's
identical.

Fill in every ``TODO`` and run ``pytest -q``. (Needs ``torch``; see the repo requirements.)
"""
from __future__ import annotations

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Save everything needed to resume training to ``path``.

    Bundle the three pieces into one dict and hand it to ``torch.save``:

        {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "epoch": epoch}

    Save the *state_dicts*, not the objects themselves — pickling the whole model hard-codes your
    class path and rots the moment you refactor.
    """
    # TODO:
    #   - build the dict above from model.state_dict(), optimizer.state_dict(), epoch
    #   - torch.save(that_dict, path)
    raise NotImplementedError


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore ``model`` and ``optimizer`` from ``path`` **in place** and return the saved epoch.

    ``load_state_dict`` mutates the object it's called on and returns a report of
    missing/unexpected keys — it does NOT return a new model. So call it on the model and optimizer
    you were passed, and return the epoch integer from the checkpoint.
    """
    # TODO:
    #   - ckpt = torch.load(path)         # you may pass weights_only=False for the optimizer state
    #   - model.load_state_dict(ckpt["model"])
    #   - optimizer.load_state_dict(ckpt["optimizer"])
    #   - return int(ckpt["epoch"])
    raise NotImplementedError


def models_allclose(model_a: torch.nn.Module, model_b: torch.nn.Module) -> bool:
    """Return True iff the two models have identical parameters.

    "Identical" means: same set of parameter names, and every corresponding tensor passes
    ``torch.allclose``. If the name sets differ, return ``False`` (don't raise).
    """
    # TODO:
    #   - grab sd_a = model_a.state_dict(), sd_b = model_b.state_dict()
    #   - if their key sets differ, return False
    #   - return True iff torch.allclose holds for every key
    raise NotImplementedError
