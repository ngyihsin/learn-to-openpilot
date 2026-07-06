# Day 46 — CLIP & Vision-Language Models

> **Week 7 · CV & VLMs** — how can a model detect "a traffic cone" from *words* (Day 45)? Because images
> and text can live in the **same vector space**. CLIP is the model that made this standard, and cosine
> similarity is how the matching works. Today you build that matching.
>
> *Gradable: assumes basic Python + Day 31 numpy.*

## Why today matters

**CLIP** (Contrastive Language-Image Pre-training) trains two encoders — one for images, one for text —
so that a picture of a dog and the phrase "a photo of a dog" map to *nearby* vectors (**embeddings**).
Once images and text share a space, you can:

- **classify with words** ("is this a cat or a dog?" → embed both captions, pick the nearer one),
- **search images by text** (and vice-versa),
- power **open-vocabulary** detection/segmentation (Day 45's Grounding DINO, and VLMs generally).

The encoders are big pretrained Transformers (Day 36 attention, scaled up). But the *comparison* — the
part that turns embeddings into a decision — is simple cosine similarity, and that's what you build.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **embedding** | A vector that represents an image or a piece of text. |
| **shared space** | Image and text embeddings live together, so they can be compared directly. |
| **cosine similarity** | Similarity by *direction*, ignoring length: 1 = same, 0 = unrelated, −1 = opposite. |
| **L2-normalize** | Rescale a vector to length 1 — so only its direction matters. |
| **zero-shot** | Classify into categories you never trained on, just by describing them in text. |

## See it with intuition

Embed the image and each candidate caption, L2-normalize them (length becomes irrelevant — only
*meaning-direction* matters), and take the cosine similarity. The caption with the highest similarity is
the model's answer. No task-specific training — that's "zero-shot."

## Do this — four steps

Work top-to-bottom in `homework.py`; run each check in this folder.

**Step 1 · `l2_normalize(x)`** — divide by `np.linalg.norm(x, axis=-1, keepdims=True)`; leave zeros alone.
```bash
python3 -c "from homework import l2_normalize as f; print(f([3,4]))"                 # expect: [0.6 0.8]
```

**Step 2 · `cosine_similarity(a, b)`** — `dot(a,b) / (|a|*|b|)`.
```bash
python3 -c "from homework import cosine_similarity as f; print(f([1,0],[0,1]), f([1,0],[-1,0]))"  # expect: 0.0 -1.0
```

**Step 3 · `similarity_matrix(image_embs, text_embs)`** — normalize both, then `img @ txt.T`.
```bash
python3 -c "import numpy as np; from homework import similarity_matrix as f; print(np.round(f([[1,0],[0,1]], [[1,0],[0,1],[1,1]]),3))"
# expect: [[1. 0. 0.707] [0. 1. 0.707]]
```

**Step 4 · `classify(image_emb, text_embs)`** — the argmax of the similarities.
```bash
python3 -c "from homework import classify as f; print(f([0.9,0.1], [[1,0],[0,1],[-1,0]]))"   # expect: 0
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 46`.

## Check yourself

- Why **L2-normalize** before comparing? What would a longer (but same-direction) vector do to a raw dot
  product, and why is that misleading?
- Cosine similarity of 0 means what about two embeddings? What does −1 mean?
- How does "embed the captions, pick the nearest" let CLIP classify categories it was never trained on
  (zero-shot)?

## Where this shows up later

This shared-space idea is the engine of the open-vocabulary models you ran on Day 45 and the VLMs your
lab researches. Week 8 has you read a paper built on exactly these ideas — and now the core operation
isn't magic, it's four functions you wrote.

**Next:** Day 47 — Datasets: KITTI & NuScenes.
