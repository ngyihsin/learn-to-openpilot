# Answer Key & Rubric — Final Exam (graders only)

Grading stance: award points for **correct substance**, not phrasing. Partial credit per the
bullet breakdowns. A factually wrong claim inside an otherwise-right answer loses the bullet
it contradicts. Code answers: logic errors lose points, syntax slips don't.

---

## Section 1 (25)

**Q1 (5).**
- (3) The five steps in order: **forward → loss → backward → update (step) → zero_grad**
  (accept zero_grad placed first — `opt.zero_grad()` before forward is the Day 23 form;
  1 pt off if steps are missing/misordered beyond that rotation).
- (1) `loss.backward()` **accumulates** into `.grad` rather than overwriting.
- (1) Forgetting it: gradients pile up across steps → effective updates grow/are stale →
  training goes haywire. (Named in the course as the #1 beginner bug.)

**Q2 (4).**
- (2) `with torch.no_grad():`
- (2) So autograd does not record the update itself into the graph (the update is not part of
  the computation being differentiated). Mentioning that tracking the update corrupts/entangles
  the next backward earns full credit.

**Q3 (4).**
- (1.5) (a) `(2, 5, 4)` — align right, pad left with 1s, dims equal-or-1 combine.
- (1) (b) Error: trailing dims 3 vs 4, neither is 1 → incompatible.
- (1.5) (c) Mean has shape `(C, 1, 1)` because `keepdim=True`; it broadcasts across H and W so
  the elementwise subtraction works.

**Q4 (4).**
- (1.5) `W` is stored as `(out_features, in_features)`.
- (1.5) `y = x @ W.T + b`
- (1) Output shape `(N, out_features)` (rule `(N, in) @ (in, out) → (N, out)`).

**Q5 (4).**
- (1.5) Model output: **raw logits** (no activation on the last layer).
- (1.5) Target: **integer class labels** (long dtype), not one-hot.
- (1) The two wrong additions: applying **softmax** yourself, and **one-hot encoding** the
  targets.

**Q6 (4).**
- (1.5) Central differences — `(f(x+eps) − f(x−eps)) / (2·eps)` per element (accept the verbal
  description "nudge both sides").
- (1) Use `float64` for accuracy.
- (1.5) Too slow because it needs on the order of **one (or two) forward passes per
  parameter** — a million-parameter model means millions of forward passes per step, vs one
  backward pass for autograd/backprop.

## Section 2 (25)

**Q7 (5).**
- (3) Shapes: conv keeps spatial size due to `padding=1` → `(N, 8, 8, 8)`; ReLU unchanged;
  MaxPool2d(2) halves → `(N, 8, 4, 4)`; flatten → `(N, 128)`; so `? = 8*4*4 = 128`.
  (Award proportionally per stage.)
- (1) #1 CNN bug: a size mismatch at the flatten→Linear boundary (wrong `in_features`).
- (1) `x.flatten(start_dim=1)`.

**Q8 (3).** Any two of (1.5 each):
- Weight sharing: the same small filter slides everywhere → far fewer parameters.
- Translation robustness: a pattern is detected regardless of position.
- Locality/hierarchy: local edge/texture detectors stack into shapes → objects.

**Q9 (4).**
- (1.5) `__len__` (number of samples) and `__getitem__(i)` (returns the `(features, label)`
  pair for sample *i*).
- (1) Batches of `[4, 4, 2]` — a ragged final batch.
- (1.5) Shuffling prevents the model from exploiting/being biased by data order (e.g. sorted
  or grouped data) — each epoch sees a different order.

**Q10 (4).**
- (1.5) AdamW **decouples weight decay** from the gradient/adaptive step; Adam mixes it into
  the gradient. Same constructor args otherwise.
- (1.5) `lr = base * min(1.0, (t + 1) / warmup_steps)` — the `+1` avoids LR 0 at step 0.
  Warmup tames the large, noisy gradients of freshly random weights.
- (1) `optimizer.param_groups[0]['lr']`.

**Q11 (5).**
- (1.5) An (ordered) dict mapping parameter/buffer **names** to their **tensors**
  (e.g. `"fc1.weight" → tensor`).
- (1.5) Pickling the whole object hard-codes the class import path — the #1 way checkpoints
  rot; a `state_dict` is plain data that loads into any correctly-shaped model.
- (1) Optimizer state (momentum buffers / Adam running averages, step counts) is needed or
  the first resumed step jolts.
- (1) It loads **in place** (mutates the model) and returns a report of missing/unexpected
  keys — not a new model.

**Q12 (4).**
- (1.5) Freezing = setting `param.requires_grad = False`; autograd then computes no gradient
  for it, so `optimizer.step()` has nothing to apply — the weight never moves.
- (1) Yes — a fresh `nn.Linear`'s parameters have `requires_grad=True` by default.
- (1.5) openpilot's driving model is a shared backbone with multiple task **heads** (path,
  lead car, lane lines); adding a capability = attaching/training a new head, not retraining
  the backbone.

## Section 3 (25)

**Q13 (5).**
- (2) An `nn.Module` is Python code tied to the training script/interpreter; a device (no
  Python, tiny runtime) needs a self-contained portable artifact capturing just the
  computation.
- (2) Tracing **runs the model once and records the ops** for that example — data-dependent
  control flow (e.g. `if x.sum() > 0:`) is baked in as whichever branch ran; other inputs
  silently take the wrong path.
- (1) `eval()` bakes in inference-time behavior of layers like dropout/batchnorm.

**Q14 (4).**
- (2) Need: the saved file + `torch.jit.load` (PyTorch runtime). Don't need: the original
  model **class definition** or training code.
- (2) Check `torch.allclose(original(x), loaded(x))` on fresh inputs — same computation.

**Q15 (6).**
- (1.5) A tiny, portable **C++ runtime** for running models on-device with minimal overhead —
  **no Python at all** at inference time.
- (2) `torch.export` the model to a clean backend-agnostic graph → compile/lower it to a
  **`.pte`** file → the ExecuTorch runtime executes it on-device.
- (1.5) A delegate offloads part/all of the graph to specialized hardware (accelerator).
- (1) Fixed graph → ahead-of-time optimization, fixed memory plan, no interpreter — predictable
  latency for real-time use.

**Q16 (4).**
- (2) `exported = torch.export.export(model, (x,))` — the example args are a **tuple**.
  (Model should be in `eval()`.) Runnable module via `exported.module()`.
- (2) The graph is specialized to the example input's **shapes** — run it with
  matching-shaped inputs afterward.

**Q17 (6).**
- (1.5) Stores weights (sometimes activations) as **8-bit integers** instead of 32-bit floats
  → ~**4×** smaller because 8 bits vs 32 bits per value.
- (1.5) `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`
  (model in eval mode; returns a quantized copy).
- (1.5) Dynamic, static, quantization-aware training (**QAT**) — QAT recovers the most
  accuracy.
- (1.5) Measure **size** (bytes) and **output error/accuracy** vs the float model (small
  relative error expected). (Accept "speed" as one of the two alongside error.)

## Section 4 (25)

**Q18 (9).** One point per stage done with a *why*; the two bold items are the discriminators:
- (1) Data: wrap frames+angles in a `Dataset`; standardize/normalize inputs; `DataLoader`
  with shuffling for training.
- (1) Split train/validation (e.g. `random_split` with a seeded generator); never evaluate on
  training data.
- (1) Model: a CNN (frames are images — local patterns, weight sharing) with a final layer
  outputting **one continuous value**.
- (**2**) Loss: a **regression** loss — **MSE / `nn.MSELoss`** (or smooth-L1/Huber) — because
  steering angle is continuous; CrossEntropy is for discrete classes and is wrong here.
- (1) Training loop: train epochs, evaluate **validation error** each epoch, keep the
  **best-val** `state_dict` (deepcopy), report val numbers honestly.
- (1) Export & lower: `eval()` → `torch.export.export(model, (example,))` → compile to
  `.pte` → ExecuTorch C++ runtime on the car computer (delegates for the accelerator).
- (1) Quantize (dynamic int8 at minimum; QAT if accuracy drop matters) → ~4× smaller, faster.
- (**1**) Verify **fidelity** (quantized/exported outputs close to the float model —
  allclose / small relative error on held-out frames) **and speed** (measure per-frame
  inference latency against the real-time frame budget on the target device).

**Q19 (6).**
- (1.5) Loop over epochs; inner loop over the DataLoader doing the five training steps.
- (1.5) Each epoch: evaluate on the validation loader; compare to best-so-far; keep the better.
- (1.5) Mode toggles: `model.train()` before the training pass, `model.eval()` before
  validation (with `torch.no_grad()` for the eval pass — award inside these 1.5).
- (1.5) The safety line: `best_state = copy.deepcopy(model.state_dict())` — a plain reference
  points at live weights and is silently overwritten next epoch. Returning/loading
  `best_state` at the end.

**Q20 (5).**
- (2) Training-set scores measure memorization, not generalization — a 100% train score tells
  you nothing about new data; val is the honest estimate.
- (1.5) The last epoch can be worse than an earlier one due to **overfitting** — val
  performance peaks and then degrades while train keeps improving.
- (1.5) The #1 evaluation bug: scoring on the **training set** and calling it "accuracy."
  (Do not accept "forgetting zero_grad" here — that is Day 22's *training* bug; the question
  asks for the evaluation bug.)

**Q21 (5).**
- (2) All development/testing happens on your computer against **recorded or simulated data**
  (replay); **never test unreviewed code on a real vehicle**. No car or comma device needed.
- (1.5) First PRs must avoid **safety-critical vehicle-control logic**; good first
  contributions: tooling, docs, tests, error messages, small bug fixes.
- (1.5) The test must **fail before the fix and pass after** it.

---

## Score sheet template

| Q | Max | S? |
|---|-----|----|
| 1 | 5 | |
| 2 | 4 | |
| 3 | 4 | |
| 4 | 4 | |
| 5 | 4 | |
| 6 | 4 | |
| 7 | 5 | |
| 8 | 3 | |
| 9 | 4 | |
| 10 | 4 | |
| 11 | 5 | |
| 12 | 4 | |
| 13 | 5 | |
| 14 | 4 | |
| 15 | 6 | |
| 16 | 4 | |
| 17 | 6 | |
| 18 | 9 | |
| 19 | 6 | |
| 20 | 5 | |
| 21 | 5 | |
| **Total** | **100** | |
