"""Day 35 homework — a 2-layer neural network with backpropagation.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

This is the big one: you build a real (tiny) neural network and train it with backpropagation — the
algorithm that computes every gradient by the chain rule, layer by layer. Day 33 did this for one
line by hand; today you do it for a network. PyTorch's autograd (Day 22) automates exactly this.

Architecture:  X --(W1,b1)--> z1 --ReLU--> h --(W2,b2)--> y ,  loss = MSE(y, target).
Shapes: X (N, Din), W1 (Din, H), b1 (H,), W2 (H, Dout), b2 (Dout,), y (N, Dout).
"""
from __future__ import annotations

import numpy as np


def relu(z):
    """ReLU activation: keep positives, clamp negatives to 0. (Hint: np.maximum.)"""
    # TODO
    raise NotImplementedError


def relu_grad(z):
    """Derivative of ReLU: 1.0 where z > 0, else 0.0 (as a float array)."""
    # TODO
    raise NotImplementedError


def forward(X, W1, b1, W2, b2):
    """Run the network. Return (y, z1, h):
        z1 = X @ W1 + b1      (hidden pre-activation)
        h  = relu(z1)         (hidden activation)
        y  = h @ W2 + b2      (output)
    You return z1 and h too because backward() needs them."""
    # TODO
    raise NotImplementedError


def mse_loss(y_pred, y_true) -> float:
    """Mean squared error over all elements. Return a plain float."""
    # TODO
    raise NotImplementedError


def backward(X, z1, h, W2, y_pred, y_true):
    """Backpropagation. Return (dW1, db1, dW2, db2). Work backwards from the loss:

        n   = y_pred.size                 # total output values (N * Dout)
        dy  = 2 * (y_pred - y_true) / n   # (N, Dout)
        dW2 = h.T @ dy                    # (H, Dout)
        db2 = dy.sum(axis=0)              # (Dout,)
        dh  = dy @ W2.T                   # (N, H)
        dz1 = dh * relu_grad(z1)          # (N, H)   <- the gradient passes back THROUGH ReLU
        dW1 = X.T @ dz1                   # (Din, H)
        db1 = dz1.sum(axis=0)             # (H,)
    """
    # TODO
    raise NotImplementedError


def train(X, y_true, W1, b1, W2, b2, lr: float = 0.05, epochs: int = 200):
    """Loop ``epochs`` times: forward, backward, then step each parameter downhill
    (param -= lr * grad). Return the final (W1, b1, W2, b2).

    Important: start by COPYING the parameters (np.array(W1, dtype=float), ...) so you don't
    modify the caller's arrays in place."""
    # TODO
    raise NotImplementedError
