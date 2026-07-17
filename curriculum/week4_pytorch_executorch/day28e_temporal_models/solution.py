"""Day 28e reference solution — temporal models."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 16
T = 4


# ----------------------------------------------------------------------------- provided --

def make_motion_dataset(n: int = 512, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    v = torch.randint(-2, 3, (n,), generator=g)
    lo = torch.clamp(2 - v * (T - 1), min=2)
    hi = torch.clamp(IMG - 3 - v * (T - 1), max=IMG - 3)
    u = torch.rand(n, generator=g)
    c0 = (lo + (hi - lo + 1).clamp(min=1) * u).long().clamp(2, IMG - 3)
    x = 0.05 * torch.randn(n, T, IMG, IMG, generator=g)
    for i in range(n):
        for t in range(T):
            c = int(c0[i] + v[i] * t)
            c = max(0, min(IMG - 2, c))
            x[i, t, :, c : c + 2] = 1.0
    return TensorDataset(x, v.float() / 2)


def mae(model: nn.Module, loader: DataLoader, last_frame_only: bool = False) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            inp = xb[:, -1:] if last_frame_only else xb
            total += (model(inp).squeeze(-1) - yb).abs().sum().item()
            count += yb.numel()
    return total / count


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 12,
    lr: float = 1e-2,
    last_frame_only: bool = False,
) -> float:
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_mae = copy.deepcopy(model.state_dict()), float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            inp = xb[:, -1:] if last_frame_only else xb
            opt.zero_grad()
            loss = criterion(model(inp).squeeze(-1), yb)
            loss.backward()
            opt.step()
        val = mae(model, val_loader, last_frame_only)
        if val < best_mae:
            best_mae = val
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_mae


# -------------------------------------------------------------------------------- solved --

def _cnn(in_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def build_single_frame_regressor() -> nn.Module:
    return _cnn(1)


def build_stacked_regressor(t: int = T) -> nn.Module:
    return _cnn(t)


class TemporalRegressor(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, d),
            nn.ReLU(),
        )
        self.cell = nn.GRUCell(d, d)
        self.head = nn.Linear(d, 1)
        self.d = d

    def init_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.d)

    def step(self, frame: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.embed(frame.unsqueeze(1))
        h_new = self.cell(e, h)
        return self.head(h_new), h_new

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        h = self.init_state(clips.shape[0])
        out = None
        for t in range(clips.shape[1]):
            out, h = self.step(clips[:, t], h)
        return out
