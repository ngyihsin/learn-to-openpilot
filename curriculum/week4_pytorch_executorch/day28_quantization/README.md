# Day 28 — Quantization & Making Models Fast

> **Week 4 · PyTorch / ExecuTorch** — 4× smaller, faster, and still good enough. Week 4 finale.

## Why today matters

A trained model's weights are 32-bit floats. On constrained hardware — a phone, the comma device —
that's often too big and too slow. **Quantization** stores weights (and sometimes activations) as
8-bit integers, cutting size roughly 4× and speeding up compute, for a small and usually acceptable
accuracy hit. It's a standard last step in the deployment pipeline you've been building all week.

Quantization comes in three flavors, in increasing order of effort **and** of accuracy recovered:
**dynamic** (weights quantized ahead of time, activations on the fly — the easiest, and what
you'll do today), **static** (also pre-computes activation scales from a small calibration set),
and **quantization-aware training (QAT)** (simulates int8 *during* training so the model learns
around the error — the most work, recovers the most accuracy). Know all three by name; reach for
the cheapest one that keeps your validation numbers.

## Learning goals

By the end you can:

- Explain what quantization trades (precision) for what (size + speed).
- Name the three flavors — dynamic, static, QAT — and say which recovers the most accuracy.
- Apply **dynamic quantization** to a model's Linear layers (int8).
- Measure a model's footprint and verify quantized outputs stay close to the float version.

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — quantize a model, plot the
   size drop, and compare fp32 vs int8 outputs to see the tiny error.
2. **Homework (~35 min).** Implement `quantize` and `model_size_bytes`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 28`. The grader checks the model
   shrinks by roughly 4× **and** that outputs stay within ~10% relative error.

## Hints

- `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` returns a
  quantized copy — quantize in `eval()` mode.
- Measure size by serializing the `state_dict` to a `BytesIO` and reading its length. The int8
  weights make the quantized `state_dict` noticeably smaller.
- Some quantization error is expected; it should be small relative to the output magnitude.

## Check yourself

- Why does int8 shrink the model ~4× vs. fp32? What's the accuracy cost, and why is it usually OK?
- Dynamic vs. static vs. quantization-aware training: what does each trade? (Dynamic is easiest;
  QAT recovers the most accuracy.)
- Why is quantization especially valuable on a battery-powered, real-time device?

## Where this shows up later

This completes the deployment story: train (Days 22–24) → feed data (25) → export (26) → lower for
on-device (27) → quantize (28). On Day 29 you meet a real model that went through this whole
pipeline to run in a car. **You've now built every piece of it.**

**Next:** Day 28b — a full training project, end-to-end. Then Day 28c trains the week's real
finale: a continuous *driving* output, verified against a frame budget.
