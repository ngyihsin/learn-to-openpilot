# Day 33 — Regression & Gradient Descent

> **Week 5 · ML & DL Foundations** — yesterday you *fit* a model with a one-line black box
> (`np.polyfit`). Today you open the box: you find the parameters yourself by walking downhill.

## Why today matters

`np.polyfit` has a closed-form answer. Almost nothing else in modern ML does. A neural network with
millions of parameters has no formula for "the best weights" — instead you start with random weights
and **repeatedly nudge them in the direction that reduces the loss.** That direction is the negative
**gradient**, and the whole procedure is **gradient descent**. It is *the* algorithm underneath every
network you'll ever train, including openpilot's driving model.

Today you implement it from scratch for the simplest model — a line, `y = w·x + b` — so the mechanics
are fully visible: predict, measure loss, compute the gradient, step. Once you've felt it on two
parameters, a million is just more of the same.

## Learning goals

By the end you can:

- Write the **MSE loss** and its **gradients** with respect to `w` and `b`, and check them against a
  finite-difference approximation.
- Explain what the **learning rate** does — and what happens if it's too big or too small.
- Implement `fit_linear` as a loop of gradient-descent steps and watch the loss fall.

## Do this

1. **Homework (~60 min).** Implement the `TODO`s in `homework.py` — five functions that build up to a
   working trainer.
2. **Grade it.** `pytest -q`  ·  or from the repo root `python tools/grade.py day 33`.

## Hints

- `predict(w, b, x)` is just `w * x + b` — vectorized over the whole array at once.
- For `mse_loss`, reuse `predict`, then average the squared errors.
- The gradients of `mean((w·x + b − y)²)`:
  - `dL/dw = mean(2 · (pred − y) · x)`
  - `dL/db = mean(2 · (pred − y))`
- A gradient-descent step moves *against* the gradient: `w_new = w − lr · dw` (same for `b`).
- `fit_linear`: start `w = 0.0, b = 0.0`, then repeat the step `epochs` times. If your loss grows
  instead of shrinks, your learning rate is too large.

## Check yourself

- Why *subtract* the gradient instead of adding it? What does the gradient point toward?
- The gradient has a factor of 2. If you dropped it, could you still converge? What would you have to
  change to compensate?
- With `w=0, b=0` on data from `y = 3x + 2`, is the first step's `dw` positive or negative? Why does
  that push `w` in the right direction?

## Where this shows up later

Day 35 stacks this idea through many layers — the gradients just get computed by **backpropagation**
instead of by hand. Day 22 (Week 4) already did this with PyTorch's autograd; today you earn the right
to trust that autograd by doing the calculus yourself.

**Next:** Day 34 — Classification: softmax & cross-entropy.
