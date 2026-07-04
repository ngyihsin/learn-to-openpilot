# Day 27 — ExecuTorch: On-Device Inference

> **Week 4 · PyTorch / ExecuTorch** — running a model where there's no Python at all.

## Why today matters

A phone, a microcontroller, or the comma device can't spin up a full PyTorch + Python stack for
every camera frame. **ExecuTorch** is PyTorch's answer: a tiny, portable C++ runtime that executes
models on-device with minimal overhead. Its pipeline is: **`torch.export`** your model to a clean,
backend-agnostic graph → compile that to a `.pte` file → run it with the ExecuTorch runtime,
optionally offloading pieces to hardware accelerators (delegates). This is essentially the path
openpilot's driving model takes from training to running in the car.

## Learning goals

By the end you can:

- Explain the ExecuTorch pipeline: export → lower/compile (`.pte`) → run on-device.
- Lower an `nn.Module` to an `ExportedProgram` with `torch.export` and run it.
- Verify the lowered graph computes the same thing as the eager model.

## Setup note

The gradeable step uses `torch.export`, which ships with modern PyTorch. The full ExecuTorch
runtime (to actually emit and run a `.pte`) installs separately and moves fast — follow the
official guide when you want to go all the way to a device:

```bash
pip install executorch     # see https://pytorch.org/executorch/ for the current instructions
```

The notebook shows the `.pte` step conceptually; the homework focuses on the export/lowering that
everything else depends on. (The grader skips if your torch lacks `torch.export`.)

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — lower a model, inspect the
   exported graph, run it, and see where the `.pte`/runtime step slots in.
2. **Homework (~35 min).** Implement `lower_model` and `run_exported`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 27`.

## Hints

- `torch.export.export(model, (example_input,))` returns an `ExportedProgram`. Note the tuple —
  it's the model's positional args.
- Put the model in `eval()` first, and get a runnable module back with `exported.module()`.
- The exported graph specializes on the example's shapes; run it with matching shapes.

## Check yourself

- Why can't you just ship the Python `nn.Module` to a microcontroller?
- What does exporting to a fixed graph give the runtime that eager mode can't (think: no Python
  interpreter, ahead-of-time optimization, fixed memory)?
- What's a "delegate," and why offload part of the graph to specialized hardware?

## Where this shows up later

Day 28 shrinks and speeds up the model with quantization — often done as part of this lowering so
the on-device model is small and fast. Together, Days 26–28 are the deployment story behind the
model you'll meet in openpilot on Day 29.

**Next:** Day 28 — Quantization & Making Models Fast.
