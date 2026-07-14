# Day 28b — A Full Training Project (End-to-End)

> **Week 4 · PyTorch / ExecuTorch** — everything this week, wired together into one real
> training run.

## Why today matters

You've built the pieces: tensors and autograd (Day 22), `nn.Module` and the training loop
(Day 23), data loading, and export. Today you assemble them into the thing a researcher
actually runs — a **complete experiment**. Not a snippet, but the full loop: build a
**Dataset**, hold out a **validation split**, train while **watching val accuracy each
epoch**, **keep the best checkpoint**, and **report a number** at the end.

This skeleton — *dataset → split → loop → validate → checkpoint → report* — is the shape of
every serious training run you'll ever write, whether it's the tiny MLP here, a CNN on
images (Week 7), or openpilot's driving model. The dataset changes; the skeleton doesn't.
Learn it once and every paper's "Experiments" section reads like something you could rebuild.

## Learning goals

By the end you can:

- Wrap tensors in a `TensorDataset` and hand them to a `DataLoader` in batches.
- Split data into **train** and **validation** with `random_split`, and say *why* you never
  score yourself on data you trained on.
- Write a training loop that validates every epoch and **checkpoints the best model**.
- Report accuracy honestly — and reload a saved `state_dict` to prove the checkpoint works.

## Do this

1. **Homework (~60 min).** Implement `make_dataset`, `split_dataset`, `build_model`,
   `evaluate`, and `train` in `homework.py`. Read each docstring — the loop's structure is
   spelled out for you; your job is to translate it into code.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 28b
   ```
   The grader checks your dataset's shapes and dtypes, that the split sizes add up, that
   training drives the loss down and clears 90% val accuracy on these (separable) blobs, and
   that loading `best_state` into a fresh model recovers that accuracy.

## Setup note

Week 4 needs PyTorch. CPU-only is fine — this whole run finishes in a couple of seconds:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- Seed a **local** generator — `g = torch.Generator().manual_seed(seed)` — and pass it into
  `torch.randn(..., generator=g)` and `random_split(..., generator=g)`. That makes runs
  reproducible without stomping on the global RNG.
- `CrossEntropyLoss` wants raw **logits** and a **long** target vector — no softmax, no
  one-hot. `build_model`'s last `Linear` already gives you logits; leave them alone.
- Day 22's #1 bug was forgetting to zero gradients. The #1 **evaluation** bug is scoring on
  the **training** set and calling it "accuracy." That number always looks great and means
  nothing. Always report **val**.
- `copy.deepcopy(model.state_dict())` when you save the best checkpoint. A plain reference
  keeps pointing at the *live* weights, so the next epoch silently overwrites your "best."
- Toggle `model.train()` before the training pass and `model.eval()` before scoring — it's
  a no-op for this MLP but a habit that saves you once dropout/batchnorm enter the picture.

## Check yourself

- Why hold out a validation set at all? What would a 100%-on-training-data model tell you?
- You keep the epoch with the best **val** accuracy, not the last epoch. When would the last
  epoch be *worse* than an earlier one? (One word: overfitting.)
- `best_state` is a `state_dict` — a plain dict of tensors, not the model. Why is saving
  *that* (rather than the whole model object) the portable way to checkpoint?

## Where this shows up later

This is the template you'll reuse for the rest of the course. Day 28c swaps this loop's
classification parts for a **continuous driving output** — a regression loss, val MAE, and a
latency check against the camera's frame budget. Then Day 29 starts the **openpilot on-ramp**,
where the same train/validate/checkpoint rhythm scales up to a real driving model.
Week 5 deepens the theory under this loop (loss landscapes, generalization); Week 7 swaps the
blobs for images and the MLP for a CNN. Every one of them is *this* skeleton with a bigger
dataset bolted on.

**Next:** Day 28c — steering-angle regression, the driving mini-project.
