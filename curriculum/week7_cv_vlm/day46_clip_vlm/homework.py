"""Day 46 homework — CLIP-style image/text matching via embeddings.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

CLIP is the idea behind open-vocabulary vision (Day 45's Grounding DINO and every VLM): an image encoder
and a text encoder map their inputs into the SAME vector space, so "a photo of a cat" lands near a cat
image. You compare them with **cosine similarity**. Here you build that comparison machinery on
pre-computed embeddings (real CLIP produces the vectors; the matching math is all yours).
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x):
    """Scale each vector to unit length (divide by its L2 norm, ``np.linalg.norm``).
    Handle a 1-D vector or the rows of a 2-D array (use axis=-1, keepdims=True).
    Leave a zero vector unchanged instead of dividing by zero."""
    # TODO
    raise NotImplementedError


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two 1-D vectors = dot(a, b) / (|a| * |b|).
    1 = same direction, 0 = orthogonal (unrelated), -1 = opposite. Return 0.0 if either is a zero vector."""
    # TODO
    raise NotImplementedError


def similarity_matrix(image_embs, text_embs):
    """Return an (n_images, n_texts) matrix of cosine similarities between every image row and every
    text row. (Hint: L2-normalize both, then one matrix multiply: img @ txt.T.)"""
    # TODO
    raise NotImplementedError


def classify(image_emb, text_embs) -> int:
    """Zero-shot classification: return the index of the text embedding most similar to the image
    (the argmax of the cosine similarities)."""
    # TODO
    raise NotImplementedError
