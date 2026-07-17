# Day 25 — Datasets, DataLoaders & Augmentation

> **Week 4 · PyTorch / ExecuTorch** — feeding the model efficiently. A coding day.

## Why today matters

A model is only as fast as the data reaching it. PyTorch splits the input pipeline into a
**`Dataset`** (how to get sample `i` and how many there are) and a **`DataLoader`** (batches,
shuffling, parallel prefetch). This separation lets you stream data too big for memory, shuffle
each epoch to avoid order bias, and keep the compute unit fed. A starved model — waiting on slow
data loading — is one of the most common and invisible training bottlenecks.

## Learning goals

By the end you can:

- Implement a custom `Dataset` (`__len__` + `__getitem__`).
- Wrap it in a `DataLoader` and reason about batch sizes (including the ragged last batch).
- Standardize inputs (zero mean, unit variance) and say why it helps training.

## Do this

1. **Homework (~45 min).** Implement `ToyDataset`, `make_loader`, and `standardize`.
2. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 25`.

## Hints

- A `Dataset` needs exactly two methods: `__len__` and `__getitem__(i)` returning `(features, label)`.
- With `batch_size=4` over 10 items you get batches of `[4, 4, 2]` — the last one is the leftover.
- `standardize`: use `X.mean(dim=0, keepdim=True)` and `X.std(dim=0, keepdim=True)`; add a small
  epsilon to the std so a constant column doesn't divide by zero.
- **Augmentation** lives in `__getitem__` too: randomly perturb the sample each time it's
  fetched (so every epoch sees a fresh variation), on the **training** set only — never the
  validation set you score yourself on. And augment the *label* when the transform demands
  it: horizontally flipping a driving frame mirrors the scene, so its steering label must be
  **negated** (Day 28c's label geometry) or you'd silently teach the model to steer the
  wrong way on half your data. Photometric tweaks (brightness, noise) leave labels alone;
  geometric ones (flips, shifts, crops) usually don't.

## Check yourself

- Why shuffle the data each epoch? What bias can sorted data introduce?
- The `DataLoader` can use `num_workers>0` to load in parallel. Which day explains why that's
  processes, not threads? (Day 11 — the GIL.)
- Why standardize inputs? What happens to gradient descent when features are on wildly different scales?

## Where this shows up later

Days 26–28 take a trained model toward deployment. A robust input pipeline like this scales from
toy tensors to the millions of logged frames used to train a real driving model.

**Next:** Day 26 — Exporting Models.
