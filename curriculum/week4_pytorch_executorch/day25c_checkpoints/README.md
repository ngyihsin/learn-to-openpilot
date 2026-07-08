# Day 25c — Saving & Loading Checkpoints

> **Week 4 · PyTorch / ExecuTorch** — `state_dict`, resuming training, and never losing a run to a crash.

## Why today matters

Real training runs take hours or days. A CV or VLM model mid-fine-tune has just spent a lot of
GPU time reaching *this* set of weights — and then the machine reboots, the job gets preempted, or
you simply want to compare epoch 7 against epoch 20. A **checkpoint** is your insurance: a file on
disk that captures everything needed to resume the run *exactly* where it stopped, or to ship the
precise weights that scored best on validation.

The thing you serialize is the model's **`state_dict`** — an ordered dictionary mapping every
parameter and buffer name (`"fc1.weight"`, `"fc1.bias"`, …) to its tensor. The optimizer has one
too (momentum buffers, step counts, the learning rate). Save both plus the epoch number and you can
reload a fresh model and keep going as if nothing happened. This is exactly how openpilot's driving
model is trained across many sessions, and how you'll resume any long research run.

## Learning goals

By the end you can:

- Explain what a `state_dict` is and why PyTorch serializes *that* instead of the whole model object.
- Write a `save_checkpoint` that bundles model + optimizer + epoch into one file with `torch.save`.
- Write a `load_checkpoint` that restores model and optimizer state **in place** and returns the
  epoch to resume from.
- Prove two models are truly identical by comparing every parameter with `torch.allclose`.

## Do this

1. **Homework (~60 min).** Implement `save_checkpoint`, `load_checkpoint`, and `models_allclose`
   in `homework.py`.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 25c
   ```
   The grader trains a small MLP, saves a checkpoint, reloads it into a *fresh* model, and confirms
   the reloaded model produces byte-for-byte identical outputs — and that the learning rate came
   back too.

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- `torch.save(obj, path)` will pickle any Python object; the convention is to save a plain `dict`
  like `{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}`.
- Load state back with `model.load_state_dict(ckpt["model"])` — note it mutates the model **in
  place** and returns a report of missing/unexpected keys, *not* a new model. Same for the optimizer.
- `torch.load` may need `weights_only=False` to un-pickle the optimizer state on newer PyTorch — the
  optimizer state isn't a pure tensor blob. (Only load files *you* wrote; un-pickling is code.)
- Save the `state_dict`, **not** the model object. Pickling the whole object hard-codes your class
  path and breaks the moment you refactor — this is the #1 way checkpoints rot.
- For `models_allclose`, iterate `state_dict()` on both models: compare the key sets first, then
  `torch.allclose` each tensor. Mismatched keys should return `False`, not raise.

## Check yourself

- Why save the `state_dict` instead of `torch.save(model, path)`? What breaks later if you pickle
  the whole object?
- Why does a resume need the **optimizer** state, not just the weights? (Think about SGD momentum or
  Adam's running averages — dropping them jolts the very next step.)
- If you load a checkpoint into a model with a *different* architecture, what does
  `load_state_dict` do, and how would you find out?

## Where this shows up later

Day 25d fine-tunes a pretrained backbone — which is just *loading someone else's checkpoint* and
continuing training on your data. Day 27 lowers a trained model to **ExecuTorch** for on-device
inference; the weights it exports are the ones you checkpointed here. Every training script you
write from now on should checkpoint early and often.

**Next:** Day 25d — Transfer Learning & Fine-Tuning.
