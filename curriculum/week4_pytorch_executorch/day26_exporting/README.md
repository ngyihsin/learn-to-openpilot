# Day 26 — Exporting Models (TorchScript / ONNX / `torch.export`)

> **Week 4 · PyTorch / ExecuTorch** — turning a training-time model into a shippable artifact.

## Why today matters

Training and deployment are different worlds. A trained `nn.Module` is Python objects and code; to
run it in a C++ service, a phone, or the comma device, you **export** it into a portable format
that captures the computation without needing your training script. TorchScript (`torch.jit`) is
the classic path; ONNX and the newer `torch.export` are others. The non-negotiable requirement:
the exported model must compute **the same thing** as the original.

## Learning goals

By the end you can:

- Export a model with `torch.jit.trace` and `save`, and reload it with `torch.jit.load`.
- Verify exported and original outputs match — the check every deployment pipeline runs.
- Explain trace vs. script vs. `torch.export`, and why "run once and record" (tracing) has limits.

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — export a model, reload it,
   and confirm identical outputs; see where tracing (which records one run) can miss data-dependent
   control flow.
2. **Homework (~40 min).** Implement `export_model` and `load_model`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 26`.

## Hints

- Call `model.eval()` before exporting so inference-time behavior (dropout, batchnorm) is baked in.
- `torch.jit.trace(model, example_input)` records the ops for that input shape; `.save(path)` writes
  the file; `torch.jit.load(path)` brings it back — **no original class definition required**.
- The grader runs both models on fresh inputs and checks `torch.allclose`.

## Check yourself

- Tracing records the operations from one example run. What kind of model logic (hint: `if x.sum()
  > 0`) can tracing silently get wrong, and how does `script`/`torch.export` help?
- Why must the exported model not depend on your Python training code?
- Why put the model in `eval()` mode before exporting?

## Where this shows up later

Day 27 goes further — lowering an exported model toward **on-device** execution (ExecuTorch).
Day 28 shrinks it with quantization. This export step is the first move in getting a model onto the car.

**Next:** Day 27 — ExecuTorch: On-Device Inference.
