# Day 24b — GPU & Mixed Precision (AMP)

> **Week 4 · PyTorch / ExecuTorch** — moving work onto your GPU, and making it faster with autocast.

## Why today matters

You have an RTX 5060 (Blackwell) sitting in your machine. So far every tensor you've made lives
on the **CPU** — fine for tiny toy problems, hopelessly slow for a real CNN or a vision-language
model. The single biggest speedup you'll ever get for free is moving the model and its data onto
the GPU, and then letting **Automatic Mixed Precision (AMP)** run the math in half precision where
it's safe to. On modern hardware AMP routinely cuts training time and memory roughly in half with
almost no code change and no accuracy loss.

Today you'll learn the two habits every serious PyTorch user has burned into their fingers:
(1) pick a device *once*, then move the model and every batch onto it, and (2) wrap the forward
pass in `torch.autocast` so the fast path is on by default. This is exactly the plumbing that
stands between "runs on my laptop CPU in an afternoon" and "trains on the GPU before lunch" — the
difference you'll feel the moment you touch real CV/VLM code or fine-tune a driving model.

## Learning goals

By the end you can:

- Choose the right `torch.device` and move an arbitrary nested structure of tensors onto it.
- Count a model's parameters (and know why *trainable* vs. *total* differ once you freeze layers).
- Run a forward pass under `torch.autocast`, and understand what mixed precision actually changes.
- Write an AMP training step, and explain where a `GradScaler` fits on a real CUDA GPU.

## Do this

1. **Homework (~60 min).** Implement the five functions in `homework.py`:
   `pick_device`, `move_to`, `count_parameters`, `autocast_forward`, and `amp_train_step`.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 24b
   ```
   The grader runs entirely on CPU, so it works even without a GPU. It checks that you pick a
   valid device, move tensors correctly through nested containers, count parameters, and that your
   AMP training step actually reduces the loss by updating the weights.

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything the grader does:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- `pick_device` is the whole "device dance" in one place: return `torch.device('cuda')` only when
  the caller wants it *and* `torch.cuda.is_available()` is true, otherwise `torch.device('cpu')`.
  Every other function just uses whatever it returns — never hard-code `'cuda'` anywhere else.
- `move_to` has to recurse: a batch is often a `dict` of tensors, or a `list`/`tuple` of them.
  Move any `torch.Tensor` with `.to(device)`, recurse into `list`/`tuple`/`dict` and **rebuild the
  same container type**, and leave everything else (ints, strings, `None`) untouched.
- **On a real CUDA GPU** you don't just call `loss.backward()` under AMP — half-precision gradients
  can underflow to zero. You wrap the backward pass with a `torch.cuda.amp.GradScaler` (newer API:
  `torch.amp.GradScaler('cuda')`): `scaler.scale(loss).backward(); scaler.step(optimizer);
  scaler.update()`. **This lesson runs on CPU**, where scaling isn't needed, so `amp_train_step`
  keeps a plain `loss.backward(); optimizer.step()`. Learn the CPU shape now; add the scaler when
  you move to the GPU.
- `torch.autocast(device_type=device.type)` is the modern, device-agnostic form — pass
  `'cpu'` or `'cuda'` from the device you picked, and the same code runs in both places.
- **Blackwell warning.** The RTX 50-series (Blackwell) needs a recent PyTorch built for **CUDA 12.8
  or newer**. An older wheel will import fine and then blow up at the first GPU op with
  `CUDA error: no kernel image is available for execution on the device` — that's a *build
  mismatch*, not a bug in your code. Install a current `cu128` (or later) wheel.

## Check yourself

- Why must the model **and** every input batch be on the *same* device? What error do you get if
  they aren't?
- What does AMP actually change — which ops run in `float16`/`bfloat16` and which stay `float32`,
  and why keep the loss and the weights in full precision?
- Why is a `GradScaler` needed on CUDA but not on CPU? What underflows without it?
- After you freeze a backbone (set `requires_grad=False`), which of your two parameter counts
  changes — total or trainable?

## Where this shows up later

Every day from here on assumes you can put a model on a device and feed it batches: Day 25's
`DataLoader` hands you batches you'll `move_to(device)`, and the CNN training in Day 26 is your
`amp_train_step` in a loop. When you fine-tune a real vision or VLM model later, AMP is what makes
it fit in your 5060's memory at all — and the device/precision discipline you build today is the
same discipline openpilot's training stack uses to get a driving model onto the GPU and, eventually,
onto the car.

**Next:** Day 25 — Datasets, DataLoaders & Augmentation.
