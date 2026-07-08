"""Day 31 homework — numpy & array thinking.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

numpy is the language the rest of this course (and every ML/CV paper) is written in. Its core idea:
an **array** holds many numbers, and math applies to the WHOLE array at once — no Python loops.
These six functions drill the operations you'll use every single day: elementwise math, broadcasting,
aggregations (mean/std), boolean masks, per-axis reductions, argmin, and reproducible randomness.
"""
from __future__ import annotations

import numpy as np


def scale_and_shift(x, a: float, b: float):
    """Return ``a * x + b`` applied to every element of ``x`` — no loop.
    (This is exactly a linear model's prediction, which you'll meet on Day 33.)"""
    # TODO: turn x into a float array, then return a * x + b
    raise NotImplementedError


def normalize(x):
    """Standardize ``x`` to mean 0, std 1: return ``(x - mean) / std``.
    (Real ML pipelines normalize inputs so features are on a comparable scale.)"""
    # TODO: use x.mean() and x.std()
    raise NotImplementedError


def select_positive(x):
    """Return only the elements of ``x`` that are strictly greater than 0,
    using a boolean mask (``x[x > 0]``)."""
    # TODO
    raise NotImplementedError


def row_means(M):
    """Given a 2-D array ``M``, return a 1-D array with the mean of each ROW.
    (Hint: reductions take an ``axis=`` argument. axis=1 collapses the columns.)"""
    # TODO
    raise NotImplementedError


def closest_index(x, value: float) -> int:
    """Return the index of the element in ``x`` nearest to ``value``.
    (Hint: np.abs to get distances, np.argmin to find the smallest.) Return a plain int."""
    # TODO
    raise NotImplementedError


def reproducible_randoms(n: int, seed: int):
    """Return ``n`` random floats in [0, 1) that are the SAME every time for a given seed.
    Use ``np.random.default_rng(seed)`` — reproducibility is what makes experiments trustworthy."""
    # TODO: make a generator from the seed, then draw n numbers with rng.random(n)
    raise NotImplementedError


def reorder(values, order):
    """Return the elements of ``values`` picked out by the integer positions in ``order`` —
    i.e. ``values[order]``. This is *fancy indexing*: indexing by a list of positions, not a
    True/False mask. (Day 32 uses exactly this to shuffle rows into train/validation splits.)"""
    # TODO: turn values into a float array, then index it with order
    raise NotImplementedError


def matmul(A, B):
    """Matrix multiply: return ``A @ B`` (as float arrays).

    The rule: an (m, k) array @ a (k, n) array gives an (m, n) array — the INNER sizes must match.
    Each output cell = a row of A dotted with a column of B. ``A.T`` (transpose) swaps rows/columns
    when you need shapes to line up. Days 35–36 (neural nets, attention) are built on exactly this."""
    # TODO: convert both to float arrays and return A @ B
    raise NotImplementedError
