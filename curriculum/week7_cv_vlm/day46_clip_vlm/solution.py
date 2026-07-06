"""Day 46 reference solution — CLIP-style image/text matching via embeddings.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained (numpy only).
CLIP maps images and text into the SAME vector space; you compare them with cosine similarity.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x):
    """Scale each vector to unit length (divide by its L2 norm). Works on a 1-D vector or the
    rows of a 2-D array. A zero vector is returned unchanged (no divide-by-zero)."""
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two 1-D vectors: 1 = same direction, 0 = orthogonal, -1 = opposite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def similarity_matrix(image_embs, text_embs):
    """Return an (n_images, n_texts) matrix of cosine similarities between every image row and
    every text row."""
    img = l2_normalize(image_embs)
    txt = l2_normalize(text_embs)
    return img @ txt.T


def classify(image_emb, text_embs) -> int:
    """Zero-shot classification: return the index of the text embedding most similar to the image."""
    sims = similarity_matrix(np.asarray(image_emb, dtype=float)[None, :], text_embs)[0]
    return int(np.argmax(sims))
