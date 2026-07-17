# Day 28d — Quantization-Aware Training from Scratch

> **Week 4 · PyTorch / ExecuTorch** — when rounding-after-training isn't good enough, teach
> the model to live with the rounding *while* it trains.

## Why today matters

Day 28 quantized a trained model and the error was tiny. That was int8 being generous. Push
harder — lower bits, smaller models, tighter accuracy targets — and **post-training
quantization (PTQ)** starts to hurt: you trained weights to fine-grained float values, then
snapped them to a coarse grid they never knew existed. On the Day 28c steering regressor at
3-bit weights, PTQ makes val MAE roughly **7× worse**.

**Quantization-aware training (QAT)** fixes this by lying to the model during training: the
forward pass uses **fake-quantized** weights (already snapped to the int grid), so the loss —
and therefore every gradient step — sees exactly the numerics the deployed model will have.
The weights *learn to be quantized*: they settle where the grid can represent them well.

There's one puzzle in the way, and it's the reason QAT feels like magic until you build it:
`round()` is a staircase. Its gradient is **zero almost everywhere**, so backprop through it
would stop learning dead. The fix is the **straight-through estimator (STE)**: use the
staircase in the forward pass, but in the backward pass pretend it was the identity and let
the gradient flow through untouched. Crude, biased — and it works. Today you implement it in
a few lines with a custom `torch.autograd.Function`, and watch QAT recover most of what PTQ
lost.

(PyTorch ships real QAT tooling — observer insertion, `prepare`/`convert` workflows, per-
channel schemes, now migrating into `torchao`. You're building the mechanism those tools
wrap, in miniature, so none of it is a black box.)

## Learning goals

By the end you can:

- Implement symmetric per-tensor **fake quantization**: scale from the tensor's max
  magnitude, round, clamp, dequantize.
- Explain why `round()` kills gradients and implement the **straight-through estimator**
  with a custom `autograd.Function`.
- Build a `QATLinear` layer and train the Day 28c road-regression task **through** the
  quantizer.
- Compare **float vs PTQ vs QAT** honestly (same data, same val MAE metric) and state when
  QAT is worth its extra cost — and why int8 PTQ was fine on Day 28.
- Place QAT in the pipeline: it happens **during training**, before export/lower (static
  quantization's calibration set, by contrast, only estimates activation ranges after).

## Do this

1. **Homework (~75 min).** Implement `quantize_tensor`, `STEQuant`, `QATLinear`,
   `ptq_quantize`, and `build_qat_regressor` in `homework.py`. The road dataset, `mae`, and
   the training loop are provided — today is about the quantizer, not the loop.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 28d
   ```
   The grader checks your quantized values sit on the right grid, that gradients actually
   flow through your STE, that PTQ at 3 bits visibly damages the regressor — and that your
   QAT model trained at the same 3 bits recovers most of that damage.

## Setup note

Week 4 needs PyTorch. CPU-only is fine — the full float/PTQ/QAT comparison runs in well
under a minute:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- Symmetric quantization to `bits`: `qmax = 2**(bits-1) - 1`, `scale = |x|.max() / qmax`,
  then `(x/scale).round().clamp(-qmax, qmax) * scale`. Compute the scale from
  `x.detach()` and clamp it away from zero (`1e-8`) so an all-zero tensor doesn't divide
  by zero.
- A custom `torch.autograd.Function` has two static methods. `forward(ctx, x, bits)`
  returns the fake-quantized tensor; `backward(ctx, grad_output)` is the STE — return the
  gradient **unchanged** for `x`, and `None` for `bits` (one return per forward input).
- `QATLinear` is `nn.Linear` with one change: `forward` uses
  `STEQuant.apply(self.weight, self.bits)` instead of raw weights, via
  `nn.functional.linear(x, w_q, self.bias)`. The *stored* weight stays float — QAT keeps
  full-precision "shadow weights" and quantizes on the fly each forward.
- `ptq_quantize` must not touch the original: `copy.deepcopy` first, then overwrite each
  Linear's weight with its quantized version under `torch.no_grad()`.
- Don't be surprised that QAT *training* loss looks jumpy — the weights hop between grid
  points. Judge it the honest way: **val MAE** of the best checkpoint, like Day 28c.
- Static quantization's "calibration set" solves a different, smaller problem: it runs
  representative inputs through the model to estimate **activation** ranges (weights need no
  data — you already have them). If you've ever picked an ADC's full-scale range, it's the
  same idea.

## Check yourself

- Why is the STE a *lie*, mathematically — and why does training still work despite it?
- At 8 bits PTQ barely hurt (Day 28); at 3 bits it was ~7× worse. What changed? (How many
  distinct values can the grid represent in each case?)
- QAT stores float shadow weights and quantizes every forward. What do you ship to the
  device — the shadow weights or the grid values?
- Where would QAT slot into the Day 28c → export → ExecuTorch pipeline, and why must it
  come *before* `torch.export` rather than after?

## Where this shows up later

This is the last stop in Week 4's deployment story: when the Day 28 shortcut isn't accurate
enough for the car, *this* is the tool that closes the gap — the same mechanism inside
`torchao`'s QAT, SNPE-style toolchains, and every "int8/int4 with no accuracy loss" paper
claim you'll read in Weeks 7–8.

**Next:** Day 28e — temporal models: one frame can't see motion; give the model memory.
