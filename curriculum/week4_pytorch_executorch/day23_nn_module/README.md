# Day 23 — `nn.Module` & the Training Loop

> **Week 4 · PyTorch / ExecuTorch** — the loop that trains every real model.

## Why today matters

Day 22 was gradient descent by hand. Real PyTorch wraps that in three abstractions: an
**`nn.Module`** owns your parameters and defines `forward`; an **optimizer** applies the update;
a **loss function** scores the output. The training loop is *the exact same five steps* — forward,
loss, backward, step, zero_grad — you just stop bookkeeping the weights yourself. Every model this
week, and openpilot's driving model, is trained by this loop.

## Learning goals

By the end you can:

- Compose an `nn.Module` (here, an MLP with `nn.Sequential`).
- Write the standard training loop with an optimizer and `CrossEntropyLoss`.
- Evaluate a classifier's accuracy with `argmax` under `no_grad`.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — train the MLP on two 2-D
   blobs, watch the loss fall, and see the learned **decision boundary** separate the classes.
2. **Homework (~50 min).** Implement `build_model`, `train`, and `accuracy`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 23`.

## Hints

- `CrossEntropyLoss` takes **raw logits** (no softmax) and **integer class labels** — don't apply
  softmax yourself.
- The five steps in order: `opt.zero_grad()` → `loss = criterion(model(X), y)` → `loss.backward()`
  → `opt.step()`. Forgetting `zero_grad` is still the classic bug (Day 22).
- `accuracy`: `model(X).argmax(dim=1)` picks the predicted class; compare to `y`, take the mean.

## Check yourself

- Why does `CrossEntropyLoss` want logits, not probabilities?
- What does the optimizer (Adam/SGD) do that you did by hand on Day 22?
- Why wrap evaluation in `torch.no_grad()`?

## Where this shows up later

Day 24 swaps the MLP for a CNN but keeps this loop. Days 26–28 take a *trained* model and export,
lower, and quantize it for deployment — the path openpilot's model takes to the car.

**Next:** Day 24 — CNNs & Computer Vision.
