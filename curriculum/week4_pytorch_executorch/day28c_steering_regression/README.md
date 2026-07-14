# Day 28c — Steering-Angle Regression: The Driving Mini-Project

> **Week 4 · PyTorch / ExecuTorch** — the true Week 4 finale: train a model that predicts a
> *continuous* driving output, then prove it's fast enough for the car.

## Why today matters

Everything you've trained so far answers "*which class?*" — horizontal or vertical, blob A or
blob B — with `CrossEntropyLoss`. But the outputs that actually drive a car are **continuous
numbers**: a steering angle, a path curvature, a distance to the lead car. openpilot's driving
model doesn't classify frames; it **regresses** quantities from them. That needs three swaps
you haven't made yet:

1. **The loss.** Continuous target → a **regression loss**: `nn.MSELoss` (mean squared error —
   the same MSE you hand-built on Day 33, now on images). Its robust cousin
   `nn.SmoothL1Loss` (Huber) is what you reach for when outliers in the labels would make
   squared error explode.
2. **The metric.** "Accuracy" is meaningless for a continuous output. You monitor **validation
   MAE** (mean absolute error) — "on average, how many units off is the steering?"
3. **The deadline.** A driving model is only useful if it answers before the next camera frame
   arrives. openpilot's camera runs at **20 Hz**, so the whole model has a **50 ms frame
   budget**. Today you *measure* inference latency instead of assuming it.

Today's task is a miniature of the real thing: synthetic "road" images containing a lane line
at some horizontal offset, labeled with the steering value that would center the car. You'll
train a small CNN to regress it, keep the best checkpoint by **val MAE**, then run the two
verifications every deployed model needs: **fidelity** (the artifact computes what the trained
model computes) and **speed** (it fits the frame budget).

## Learning goals

By the end you can:

- Build a CNN whose head outputs **one continuous value** and train it with `nn.MSELoss`.
- Explain when you'd prefer `nn.SmoothL1Loss` (Huber) over plain MSE.
- Evaluate a regressor honestly with **validation MAE**, and checkpoint the best epoch by it.
- **Measure inference latency** (warmup, then time many runs, report the median) and check it
  against a frame budget in FPS.

## Do this

1. **Homework (~60 min).** Implement `build_regressor`, `mae`, `train_regressor`,
   `measure_latency_ms`, and `meets_frame_budget` in `homework.py`. The synthetic road
   dataset (`make_road_dataset`) is provided — read it first so you know what the model sees.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 28c
   ```
   The grader checks your model trains to a low validation MAE, that the best checkpoint
   reloads to the same score, that your latency number is a real measurement, and that the
   frame-budget check does the right arithmetic.

## Setup note

Week 4 needs PyTorch. CPU-only is fine — this trains in seconds:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- **The #1 regression bug is a shape bug.** Your model outputs `(N, 1)` but the targets are
  `(N,)`. Feed those to `MSELoss` and broadcasting quietly turns the difference into an
  `(N, N)` matrix — PyTorch even warns you. `squeeze(-1)` the prediction (or `unsqueeze` the
  target) so both sides are `(N,)` **before** the loss.
- Regression targets are **floats** — no `long()`, no one-hot. That's the mirror image of the
  `CrossEntropyLoss` rules you learned on Day 23.
- The training loop is Day 28b's skeleton verbatim — only the loss (`nn.MSELoss`), the metric
  (MAE, lower is better), and the head (one output) change. `copy.deepcopy` the best
  `state_dict`, toggle `model.train()` / `model.eval()`, evaluate under `torch.no_grad()`.
- For latency: run a few **warmup** passes first (the first calls pay one-time costs), then
  time each of many runs with `time.perf_counter()` and report the **median** — it ignores
  the occasional OS hiccup that would pollute a mean. Milliseconds = seconds × 1000.
- A frame budget is just arithmetic: at `fps` frames per second you have `1000 / fps`
  milliseconds per frame. The model meets the budget when its latency is ≤ that.

## Check yourself

- Why would training with `CrossEntropyLoss` be *impossible* here, not just wrong? (What
  would the "classes" even be?)
- Your val MAE is 0.03 and steering is in [-1, 1]. Is that good? What real-world information
  would you need before saying "good enough to ship"?
- Why measure latency on the *deployed* artifact (exported + quantized, on the target device)
  rather than on the eager model on your laptop?
- Why median latency and not mean? When would you also care about the **worst case** (tail
  latency) in a car?

## Where this shows up later

This is the last new skill of Week 4 — you now have the *complete* path: continuous driving
output → regression training → best-val checkpoint → export (Day 26) → lower (Day 27) →
quantize (Day 28) → **verify fidelity and latency** (today). Day 29 walks into openpilot,
where `modeld` runs exactly this kind of model against exactly this kind of deadline, 20
times a second.

**Next:** Day 29 — the openpilot on-ramp.
