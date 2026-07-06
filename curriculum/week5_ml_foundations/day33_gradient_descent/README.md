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

### What is a gradient? (no calculus needed)

Imagine standing on a hillside in fog, trying to reach the valley floor. You can't see the bottom, but
you can *feel the slope under your feet*. The **gradient** is exactly that feeling: for each knob
(`w`, `b`), a number saying *"if you nudge this knob up a little, does the loss go up or down — and how
steeply?"* Positive gradient = increasing the knob increases the loss, so step it **down**. That's the
whole idea; "derivative" and "partial derivative" are just the formal names for slope-per-knob.

Where do the formulas below come from? Calculus mechanically derives them (the `2` comes from the
derivative of the square in MSE). **You don't need to derive them today — take them on trust and
implement them.** The grader includes a *finite-difference check*: it wiggles `w` by a tiny amount,
measures how the loss actually changes, and confirms your formula matches reality. That check is how
practitioners verify gradients without trusting anyone — including themselves.

## Learning goals

By the end you can:

- Write the **MSE loss** and its **gradients** with respect to `w` and `b`, and check them against a
  finite-difference approximation.
- Explain what the **learning rate** does — and what happens if it's too big or too small.
- Implement `fit_linear` as a loop of gradient-descent steps and watch the loss fall.

## Do this — five small steps

0. **Concept + visualization (~10 min).** Open `lesson.ipynb` and run every cell — watch the loss fall
   and the line snap onto the data. Then build the machinery yourself.

Work top-to-bottom in `homework.py`; after each function, run its check in this folder.

**Step 1 · `predict(w, b, x)`** — just `w * x + b`, vectorized over the whole array (Day 31 Step 1!).
```bash
python3 -c "from homework import predict as f; print(f(2.0, 1.0, [0,1,2]))"          # expect: [1. 3. 5.]
```

**Step 2 · `mse_loss(w, b, x, y)`** — reuse `predict`, then average the squared errors (Day 32's MSE).
```bash
python3 -c "from homework import mse_loss as f; print(f(1.0, 0.0, [1,2], [0,0]))"    # expect: 2.5
```
*(pred = [1, 2]; squared misses 1 and 4; mean 2.5.)*

**Step 3 · `gradients(w, b, x, y)`** — the two slope formulas, exactly as given:
`dw = mean(2·(pred−y)·x)` and `db = mean(2·(pred−y))`. Return them as a tuple.
```bash
python3 -c "from homework import gradients as f; print(f(0.0, 0.0, [1,2], [3,6]))"   # expect: (-15.0, -9.0)
```
*(Both negative — the loss goes DOWN if w and b go up. Which is right: the data `y=3x` needs w>0.)*

**Step 4 · `gradient_descent_step(w, b, x, y, lr)`** — move *against* the slope: `w − lr·dw`, `b − lr·db`.
```bash
python3 -c "from homework import gradient_descent_step as f; print(f(0.0, 0.0, [1,2], [3,6], 0.1))"   # expect: (1.5, 0.9)
```

**Step 5 · `fit_linear(x, y, lr, epochs)`** — start `w = b = 0.0`, repeat Step 4 `epochs` times.
```bash
python3 -c "from homework import fit_linear as f; w,b=f([0,1,2,3],[2,5,8,11],lr=0.05,epochs=3000); print(round(w,2), round(b,2))"   # expect: 3.0 2.0
```
*(The data is y = 3x + 2 — gradient descent recovers the 3 and the 2.)* If the loss ever grows instead
of shrinking, your learning rate is too large.

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 33`.

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
