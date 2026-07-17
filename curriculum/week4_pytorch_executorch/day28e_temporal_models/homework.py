"""Day 28e homework — temporal models: seeing motion.

A single frame contains zero information about a *rate* (speed, drift, closing distance) —
two scenes with different speeds can produce identical frames. Today you prove that with a
model that stays blind no matter how it trains, then fix it twice: **frame stacking** (T
frames as T input channels) and **recurrent state** (a `nn.GRUCell` carrying memory, with
the state as an explicit input/output so the model stays an exportable pure function).

The motion dataset, ``mae``, and the training loop are provided — your job is the models.

Fill in every ``TODO`` and run ``pytest -q``.  (Needs torch.)
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMG = 16  # frames are IMG x IMG
T = 4     # frames per clip


# ----------------------------------------------------------------------------- provided --

def make_motion_dataset(n: int = 512, seed: int = 0) -> TensorDataset:
    """PROVIDED — clips of T frames: a 2-px bright bar starts at column c0 and slides
    sideways at a constant speed v (pixels/frame). The label is v/2 in {-1,-0.5,0,0.5,1}.

    The *sample* is the clip `(T, IMG, IMG)` — the unit the DataLoader shuffles. Frame order
    inside a clip is the signal; only the clip's *identity* may be shuffled.
    """
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
    """PROVIDED — val MAE; slices the last frame (channel dim kept) for single-frame models."""
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
    """PROVIDED — Day 28c's loop (MSE + Adam + best-val checkpoint), clip-aware."""
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


# --------------------------------------------------------------------------- your TODOs --

def build_single_frame_regressor() -> nn.Module:
    """Day 28c's CNN regressor, unchanged: input `(B, 1, IMG, IMG)`, one output.

        Conv2d(1, 8, 3, padding=1) -> ReLU -> MaxPool2d(2)
        -> Flatten -> Linear(8*8*8, 32) -> ReLU -> Linear(32, 1)

    This model is doomed on today's task — build it anyway; the grader uses its failure
    as the proof that the information isn't in a single frame.
    """
    # TODO
    raise NotImplementedError


def build_stacked_regressor(t: int = T) -> nn.Module:
    """Frame stacking: the same CNN, but the first conv reads `t` channels — one per frame.

    A `(B, T, IMG, IMG)` clip is already the `(B, C, H, W)` layout a CNN expects; time
    simply becomes channels, and layer one can subtract frames to see motion.
    """
    # TODO: identical to build_single_frame_regressor() except Conv2d(t, 8, ...).
    raise NotImplementedError


class TemporalRegressor(nn.Module):
    """A streaming model: embed each frame, carry memory in a GRU cell, predict from memory.

    Deployment-honest design (Day 27): the hidden state is an EXPLICIT argument and return
    value of `step` — never stored on `self` — so the model remains a pure function that
    `torch.export` can capture, with the runtime round-tripping `h` between 20 Hz calls.
    """

    def __init__(self, d: int = 32):
        super().__init__()
        # TODO:
        #   self.embed = Day 24's CNN minus the head:
        #       Conv2d(1, 8, 3, padding=1) -> ReLU -> MaxPool2d(2)
        #       -> Flatten -> Linear(8*8*8, d) -> ReLU
        #   self.cell = nn.GRUCell(d, d)
        #   self.head = nn.Linear(d, 1)
        #   self.d = d
        raise NotImplementedError

    def init_state(self, batch: int) -> torch.Tensor:
        """The 'new drive' state: zeros of shape (batch, d). Fresh drive, fresh zeros —
        stale state from the last route is a wrong prior about this one."""
        # TODO
        raise NotImplementedError

    def step(self, frame: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One 20 Hz tick: `(frame (B, IMG, IMG), h (B, d)) -> (prediction (B, 1), h_new)`.

        unsqueeze the frame to (B, 1, IMG, IMG) for the embed CNN, advance the GRU cell,
        predict from the NEW state, and return both. No side effects.
        """
        # TODO
        raise NotImplementedError

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Offline/training path over `(B, T, IMG, IMG)`: init_state, thread `step` across
        the T frames in order, return the final prediction `(B, 1)`."""
        # TODO
        raise NotImplementedError
