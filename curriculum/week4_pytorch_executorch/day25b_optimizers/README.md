# Day 25b — Optimizers & Learning-Rate Scheduling

> **Week 4 · PyTorch / ExecuTorch** — SGD vs Adam, weight decay, and shaping the learning rate over time.

## Why today matters

When you sit down to reproduce a paper, the architecture is usually spelled out for you. What is
*not* spelled out — or is buried in an appendix — is the **training recipe**: which optimizer,
what learning rate, whether it warms up, and how it decays. Get that wrong and the exact same
model either diverges to NaN or crawls to a mediocre score. The optimizer and its learning-rate
schedule are the knobs you tune most.

openpilot's driving model, every CV backbone, every vision-language model you'll read this
summer — all of them ship a specific recipe. Today you'll build the two pieces of that recipe by
hand: choosing an optimizer (SGD, Adam, AdamW) with weight decay, and computing how the learning
rate should change step by step. After this you'll recognize a training config on sight.

## Learning goals

By the end you can:

- Build any of `SGD`, `Adam`, `AdamW` over a model's parameters, with a learning rate and weight
  decay, and say when you'd reach for each.
- Explain **weight decay** in one sentence (a pull toward smaller weights = regularization).
- Compute a **step-decay** and a **linear-warmup** learning rate by hand.
- Read a learning rate back off a live optimizer and drive a `torch.optim.lr_scheduler`.

## Do this

1. **Homework (~60 min).** Implement `make_optimizer`, `step_decay_lr`, `warmup_lr`, and
   `lr_schedule` in `homework.py`. Fill in every `TODO`.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 25b
   ```
   The grader checks you build the right optimizer type, set lr/weight-decay correctly, and that
   your schedule math matches the exact expected numbers.

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- An optimizer is built over an *iterator* of parameters: `torch.optim.Adam(model.parameters(),
  lr=..., weight_decay=...)`. Pass `name.lower()` through an if/elif and `raise ValueError` on the
  fallthrough — an unknown optimizer name should fail loudly, not silently pick a default.
- **`AdamW` vs `Adam`** differ in *how* weight decay is applied (AdamW decouples it from the
  gradient step). For this exercise both take the same constructor arguments — the difference is
  the class you instantiate.
- `step_decay_lr` is just `base_lr * (drop ** (step // every))`. Integer division `//` is the
  whole trick: the lr holds flat for `every` steps, then drops by a factor of `drop`.
- `warmup_lr` ramps linearly from a small value up to `base_lr` over `warmup_steps`, then holds:
  `base_lr * min(1.0, (step + 1) / warmup_steps)`. The `+1` means step 0 isn't a dead zero.
- In `lr_schedule`, read the lr *before* stepping the scheduler: `optimizer.param_groups[0]['lr']`.
  A scheduler mutates that value in place when you call `scheduler.step()`.

## Check yourself

- Why does Adam often "just work" at a learning rate where plain SGD stalls or diverges? (Think
  about per-parameter adaptive step sizes.)
- What problem does **warmup** solve at the very start of training, when the weights are random
  and the gradients are large and noisy?
- If you double the batch size, which way does the learning rate usually need to move, and why?
- Weight decay and an L2 penalty on the weights look almost identical for SGD — so why does
  AdamW exist?

## Where this shows up later

Day 25c saves and loads the optimizer state alongside the model so you can *resume* training
exactly where you left off — schedulers carry state too. Every training script in Weeks 5–8, and
openpilot's own model training, is `make_optimizer` + a schedule wrapped around the Day 23
training loop. Reading a paper's "we train with AdamW, lr 3e-4, cosine decay, 500 warmup steps"
will now parse as a concrete thing you can build.

**Next:** Day 25c — Saving & Loading Checkpoints.
