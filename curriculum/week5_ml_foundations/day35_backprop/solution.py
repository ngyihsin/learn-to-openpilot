"""Day 35 reference solution — a 2-layer neural network with backpropagation.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained.

Architecture:  X --(W1,b1)--> z1 --ReLU--> h --(W2,b2)--> y ,  loss = MSE(y, target).
Shapes: X (N, Din), W1 (Din, H), b1 (H,), W2 (H, Dout), b2 (Dout,), y (N, Dout).
"""
from __future__ import annotations

import numpy as np


def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_grad(z):
    # derivative of ReLU: 1 where the input was positive, else 0
    return (np.asarray(z, dtype=float) > 0).astype(float)


def forward(X, W1, b1, W2, b2):
    """Return (y, z1, h): the output plus the intermediates backward() needs."""
    X = np.asarray(X, dtype=float)
    z1 = X @ W1 + b1        # pre-activation of hidden layer
    h = relu(z1)            # hidden activations
    y = h @ W2 + b2         # linear output
    return y, z1, h


def mse_loss(y_pred, y_true) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def backward(X, z1, h, W2, y_pred, y_true):
    """Return (dW1, db1, dW2, db2): gradients of the MSE loss w.r.t. each parameter."""
    X = np.asarray(X, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    n = y_pred.size                     # total number of output values (N * Dout)

    dy = 2.0 * (y_pred - y_true) / n    # dL/dy        (N, Dout)
    dW2 = h.T @ dy                      # dL/dW2       (H, Dout)
    db2 = dy.sum(axis=0)               # dL/db2       (Dout,)
    dh = dy @ W2.T                      # dL/dh        (N, H)
    dz1 = dh * relu_grad(z1)           # through ReLU (N, H)
    dW1 = X.T @ dz1                     # dL/dW1       (Din, H)
    db1 = dz1.sum(axis=0)             # dL/db1       (H,)
    return dW1, db1, dW2, db2


def train(X, y_true, W1, b1, W2, b2, lr: float = 0.05, epochs: int = 200):
    """Run forward → backward → gradient-descent update ``epochs`` times. Return final params."""
    W1, b1, W2, b2 = (np.array(p, dtype=float) for p in (W1, b1, W2, b2))
    for _ in range(epochs):
        y, z1, h = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, z1, h, W2, y, y_true)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return W1, b1, W2, b2
