"""Day 28d homework — quantization-aware training, built from scratch.

Day 28's post-training quantization (PTQ) rounds a *finished* model onto the int grid — fine
at int8, painful at low bits. **QAT** instead fake-quantizes the weights during every forward
pass of training, so the loss (and every gradient step) sees deployed-model numerics and the
weights learn to sit where the grid can represent them.

The obstacle: ``round()`` has zero gradient almost everywhere. The **straight-through
estimator (STE)** uses the staircase forward and pretends it's the identity backward — you'll
implement it as a custom ``torch.autograd.Function``.

The road dataset (Day 28c's, flattened for an MLP), ``mae``, and the training loop are
provided. Your job is the quantizer.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 16  # images are 16x16, flattened to 256 features


# ----------------------------------------------------------------------------- provided --

def make_road_dataset(n: int = 512, seed: int = 0) -> TensorDataset:
    """PROVIDED — Day 28c's synthetic road frames, flattened for an MLP regressor."""
    g = torch.Generator().manual_seed(seed)
    cols = torch.randint(2, IMG - 3, (n,), generator=g)
    x = 0.05 * torch.randn(n, 1, IMG, IMG, generator=g)
    for i, c in enumerate(cols):
        x[i, 0, :, c : c + 2] = 1.0
    center = (IMG - 1) / 2
    y = (cols.float() - center) / center
    return TensorDataset(x.flatten(1), y)


def mae(model: nn.Module, loader: DataLoader) -> float:
    """PROVIDED — mean absolute error, the Day 28c way."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            total += (model(xb).squeeze(-1) - yb).abs().sum().item()
            count += yb.numel()
    return total / count


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 12,
    lr: float = 1e-2,
) -> float:
    """PROVIDED — Day 28c's loop: MSE + Adam, keep and load the best-val-MAE state.

    Works unchanged for float models AND QAT models — that's the point: QAT is the same
    training loop with quantization-aware layers inside.
    """
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_mae = copy.deepcopy(model.state_dict()), float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        val = mae(model, val_loader)
        if val < best_mae:
            best_mae = val
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_mae


# --------------------------------------------------------------------------- your TODOs --

def quantize_tensor(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor fake quantization: snap `x` onto a `bits`-bit signed grid.

    Recipe (see Hints):
      qmax  = 2**(bits-1) - 1                       # e.g. 127 for int8, 3 for 3-bit
      scale = x.detach().abs().max() / qmax         # clamp(min=1e-8): no divide-by-zero
      return (x/scale).round().clamp(-qmax, qmax) * scale

    The result is float dtype but holds only 2*qmax + 1 distinct values ("fake" quant:
    real int storage happens at export; training only needs the *numerics*).
    """
    # TODO
    raise NotImplementedError


class STEQuant(torch.autograd.Function):
    """Fake-quantize forward; straight-through (identity) backward.

    `round()`'s true gradient is zero almost everywhere — backprop through it would stop
    learning dead. The STE forwards the staircase but backwards the identity.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        # TODO: return quantize_tensor(x, bits)
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: pass the gradient straight through for `x`; return None for `bits`.
        raise NotImplementedError


class QATLinear(nn.Linear):
    """An nn.Linear that trains through fake-quantized weights.

    Storage stays full-precision ("shadow weights"); every forward quantizes on the fly:
        w_q = STEQuant.apply(self.weight, self.bits)
        return nn.functional.linear(x, w_q, self.bias)
    (Bias stays float — it's tiny and standard practice keeps it full precision here.)
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__(in_features, out_features)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def ptq_quantize(model: nn.Module, bits: int = 8) -> nn.Module:
    """Post-training quantization, Day 28 style but by hand: return a deepcopy of `model`
    with every nn.Linear weight snapped to the grid. Never mutate the original; do the
    overwrite under torch.no_grad()."""
    # TODO
    raise NotImplementedError


def build_qat_regressor(bits: int = 8) -> nn.Module:
    """The Day 28c-shaped MLP regressor, but quantization-aware:
        QATLinear(IMG*IMG, 32, bits) -> ReLU -> QATLinear(32, 1, bits)
    """
    # TODO
    raise NotImplementedError
