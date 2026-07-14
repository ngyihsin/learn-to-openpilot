"""Day 28c homework — steering-angle regression: the driving mini-project.

The models you've trained so far pick a *class*. A driving model predicts *numbers*: steering
angle, path curvature, lead-car distance. That means a **regression loss** (`nn.MSELoss` — or
`nn.SmoothL1Loss` when label outliers would make squared error explode), a regression metric
(**validation MAE**, lower is better), and one more thing no classifier lesson needed: a
**latency measurement**, because a model that misses the camera's frame budget is useless in
a car no matter how accurate it is.

The dataset below is provided: 16×16 grayscale "road" images with a bright lane line at some
horizontal offset; the label is the steering value in [-1, 1] that would center the car on it.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 16  # images are (1, IMG, IMG)


def make_road_dataset(n: int = 512, seed: int = 0) -> TensorDataset:
    """PROVIDED — do not edit. Synthetic road frames + steering labels.

    Each sample: a noisy grayscale image with a bright 2-px vertical lane line at column
    ``c``; the label is the steering value ``(c - center) / center`` in roughly [-0.75, 0.75].
    A local, seeded generator keeps runs reproducible (the Day 28b habit).
    """
    g = torch.Generator().manual_seed(seed)
    cols = torch.randint(2, IMG - 3, (n,), generator=g)
    x = 0.05 * torch.randn(n, 1, IMG, IMG, generator=g)
    for i, c in enumerate(cols):
        x[i, 0, :, c : c + 2] = 1.0
    center = (IMG - 1) / 2
    y = (cols.float() - center) / center
    return TensorDataset(x, y)


def build_regressor() -> nn.Module:
    """A small CNN that maps a (N, 1, 16, 16) batch to ONE continuous value per image.

    Suggested architecture (Day 24's shape arithmetic applies):
        Conv2d(1, 8, kernel_size=3, padding=1) -> ReLU -> MaxPool2d(2)   # (N, 8, 8, 8)
        -> Flatten                                                       # (N, 8*8*8)
        -> Linear(8*8*8, 32) -> ReLU -> Linear(32, 1)                    # (N, 1)

    The head is Linear(..., 1) — a regressor's "logits" are just the number itself.
    """
    # TODO: return the nn.Sequential above.
    raise NotImplementedError


def mae(model: nn.Module, loader: DataLoader) -> float:
    """Mean absolute error of `model` over `loader`, as a plain float.

    Evaluation discipline is the same as Day 28b: `model.eval()`, `torch.no_grad()`.
    Mind the shapes: predictions come out (B, 1), targets are (B,) — squeeze before
    subtracting or broadcasting silently gives you a (B, B) matrix of nonsense.
    """
    # TODO: sum |pred - y| over all samples, divide by the dataset size.
    raise NotImplementedError


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 12,
    lr: float = 1e-2,
) -> tuple[dict, float]:
    """Train with nn.MSELoss + Adam; return (best_state, best_val_mae).

    Day 28b's skeleton with the classification parts swapped out:
      - loss: nn.MSELoss() on squeezed (B,) predictions vs float (B,) targets
      - each epoch: train over train_loader (the five steps), then compute val MAE
      - keep the best (lowest) val MAE seen and a `copy.deepcopy` of that state_dict
      - toggle model.train() for the training pass (mae() handles eval mode)
    """
    # TODO: implement the loop; return (best_state, best_val_mae).
    raise NotImplementedError


def measure_latency_ms(
    model: nn.Module, example: torch.Tensor, warmup: int = 5, iters: int = 50
) -> float:
    """Median per-inference latency in milliseconds for `model(example)`.

    The honest way to time a model:
      1. `model.eval()` and run everything under `torch.no_grad()` — we time inference,
         not autograd bookkeeping.
      2. Run `warmup` untimed passes first (first calls pay one-time setup costs).
      3. Time each of `iters` passes with `time.perf_counter()` and return the **median**
         in ms — the median shrugs off the occasional OS hiccup that would skew a mean.
    """
    # TODO: warmup, time each iteration, return median * 1000.
    raise NotImplementedError


def meets_frame_budget(latency_ms: float, fps: float) -> bool:
    """Does a model with this per-frame latency keep up with a camera at `fps`?

    At `fps` frames per second the budget is (1000 / fps) ms per frame; the model must
    answer in that time or less. openpilot's cameras run at 20 Hz -> a 50 ms budget.
    """
    # TODO: one line of arithmetic.
    raise NotImplementedError
