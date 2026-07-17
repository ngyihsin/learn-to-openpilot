"""Day 28d reference solution — QAT from scratch."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 16


# ----------------------------------------------------------------------------- provided --

def make_road_dataset(n: int = 512, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    cols = torch.randint(2, IMG - 3, (n,), generator=g)
    x = 0.05 * torch.randn(n, 1, IMG, IMG, generator=g)
    for i, c in enumerate(cols):
        x[i, 0, :, c : c + 2] = 1.0
    center = (IMG - 1) / 2
    y = (cols.float() - center) / center
    return TensorDataset(x.flatten(1), y)


def mae(model: nn.Module, loader: DataLoader) -> float:
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


# -------------------------------------------------------------------------------- solved --

def quantize_tensor(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


class STEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        return quantize_tensor(x, bits)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class QATLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__(in_features, out_features)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = STEQuant.apply(self.weight, self.bits)
        return nn.functional.linear(x, w_q, self.bias)


def ptq_quantize(model: nn.Module, bits: int = 8) -> nn.Module:
    q = copy.deepcopy(model)
    with torch.no_grad():
        for m in q.modules():
            if isinstance(m, nn.Linear):
                m.weight.copy_(quantize_tensor(m.weight, bits))
    return q


def build_qat_regressor(bits: int = 8) -> nn.Module:
    return nn.Sequential(
        QATLinear(IMG * IMG, 32, bits),
        nn.ReLU(),
        QATLinear(32, 1, bits),
    )
