# Day 24 — CNNs & Computer Vision

> **Week 4 · PyTorch / ExecuTorch** — models that *see*.

## Why today matters

Convolutional Neural Networks are how computers understand images. Instead of flattening pixels
(which throws away spatial layout), a CNN slides small learnable **filters** across the image to
detect local patterns — edges, corners, textures — no matter where they appear. Stack a few and
the network builds up from edges to shapes to objects. This is the core of openpilot's vision
model, which turns camera frames into a driving path.

Today you build a tiny CNN and train it to distinguish "bright row" from "bright column" images —
a spatial task an MLP fumbles but convolution is built for.

## Learning goals

By the end you can:

- Explain what a convolution filter does and why it beats a dense layer on images.
- Build a `Conv2d → ReLU → MaxPool → Flatten → Linear` classifier and get the shapes right.
- Reuse the Day 23 training loop on image data.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — see sample images, watch
   accuracy climb, and visualize what the learned filters respond to.
2. **Homework (~55 min).** Implement `SmallCNN` (layers + `forward`), `build_cnn`, and `train`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 24`.

## Hints

- **Shapes are the whole battle.** `Conv2d(1, 8, 3, padding=1)` keeps 8×8; `MaxPool2d(2)` halves
  it to 4×4; so after the conv stack you have `8 channels × 4 × 4 = 128` features. That's the
  input size of the final `Linear`. Mismatch here is the #1 CNN bug.
- `x.flatten(start_dim=1)` flattens everything except the batch dimension.
- Same training loop as Day 23 — CrossEntropyLoss + Adam.

## Check yourself

- Why does a convolution share the *same* filter weights across every position? What does that buy
  you (parameters, translation invariance)?
- If you removed `padding=1`, what would the spatial size become after the conv, and why?
- Why is a CNN a better fit than an MLP for "is the bright line horizontal or vertical?"

## Where this shows up later

Day 25 feeds images efficiently with `Dataset`/`DataLoader`; Days 26–28 export and quantize a
trained model for deployment. A CNN like this — much bigger — is what runs on the comma device.

**Next:** Day 25 — Datasets, DataLoaders & Augmentation.
