# Day 22 — Tensors & Autograd

> **Week 4 · PyTorch / ExecuTorch** — the foundation everything else this week stands on.
>
> **New to arrays, gradients, or machine learning?** This day assumes them. The gentler on-ramp is
> **Week 5 first**: Day 31 (numpy arrays — a *tensor* is numpy's array idea, on a GPU with gradients),
> Day 32 (model/loss/generalization), and Day 33 (gradient descent by hand, including *what a gradient
> is* in plain words). Do those, then return here — autograd will feel like automation of things you
> already built. See the study-order note in `SYLLABUS.md`.

## Why today matters

Every neural network — including the one driving a comma device in openpilot — is trained the
same way: define a computation, measure how wrong it is, and nudge the parameters *downhill*
along the gradient of the error. PyTorch's **autograd** computes those gradients automatically.
It can feel like magic. Today you'll pull back the curtain and prove it's just calculus: you'll
compute a gradient the slow, obviously-correct numerical way, and watch autograd give the same
answer. Then you'll train a model — a straight line — with a training loop you write by hand.

Master this loop and *every* PyTorch model you meet later is a variation on it.

## Learning goals

By the end you can:

- Create tensors, use `requires_grad`, and read `.grad` after `.backward()`.
- Explain what a gradient *is* and check one numerically (central differences).
- Write the canonical training loop: **forward → loss → backward → update → zero_grad** — and
  say why each line is there (especially why you must zero the gradients).

## Do this

1. **Concept + visualization (~25 min).** Run the notebook:
   ```bash
   jupyter lab lesson.ipynb
   ```
   You'll watch a line fit itself to noisy data and see the **loss curve** fall epoch by epoch.

2. **Homework (~60 min).** Implement `numerical_gradient` and `fit_line` in `homework.py`.

3. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 22
   ```
   The grader confirms your numerical gradient matches PyTorch's autograd, and that your
   training loop actually recovers the true slope and intercept.

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- `loss.backward()` *accumulates* into `.grad`. If you forget `zero_()`, your gradients pile up
  across epochs and training goes haywire — this is the #1 beginner bug. The grader will catch it.
- Update weights inside `with torch.no_grad():` so the update step itself isn't recorded by
  autograd.
- For `numerical_gradient`, use `float64` for accuracy and central differences (both sides),
  not a one-sided difference.
- Numerical gradients are for *checking*, never for training: central differences cost two
  forward passes **per parameter**, so a million-parameter model would need millions of forward
  passes for a single update. One `backward()` gets every gradient at once — that's the entire
  reason autograd (and backprop) exists.
- You'll sometimes see the loop written with the zero step first — `opt.zero_grad()` before the
  forward pass, as Day 23 does. It's the same loop rotated: all that matters is that gradients
  are zeroed before the next `backward()`.

## Check yourself

- Why is the numerical gradient too slow to use for real training, even though it's correct?
  (Think about how many forward passes a million-parameter model would need.)
- What does the learning rate control? What happens if it's way too big?
- Why zero the gradients every step instead of every N steps?

## Where this shows up later

Day 23 replaces your hand-written loop with `nn.Module` and an optimizer, but the five steps
are identical. Days 24–26 scale it up to CNNs and export; Day 27 lowers a trained model to
**ExecuTorch** to run on-device — the same path openpilot's driving model takes from training
to the car.

**Next:** Day 23 — `nn.Module` & the training loop.
