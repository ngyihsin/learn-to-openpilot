# Final Exam — Answers (Student S1)

## Q1 — The canonical training loop

The five named steps, in order:

1. **Forward** — run the model on the batch: `pred = model(X)`
2. **Loss** — score the output: `loss = criterion(pred, y)`
3. **Backward** — compute gradients: `loss.backward()`
4. **Update** — apply the step: `opt.step()` (or `w -= lr * w.grad` by hand)
5. **Zero grad** — clear the gradients: `opt.zero_grad()` (or `w.grad.zero_()`)

(In real code the `zero_grad()` is usually placed at the top of the loop before the forward
pass — the cycle is the same either way.)

(a) Gradients must be zeroed every step because `loss.backward()` **accumulates** into each
parameter's `.grad` attribute — it adds to whatever is already there; it does not overwrite.

(b) The mechanism of failure: if you never zero, each step's `.grad` becomes the **sum of all
past steps' gradients**, not the gradient of the current batch. The effective step is therefore
computed from a stale, ever-growing accumulated gradient — the updates grow and point in wrong
directions, and training goes haywire (loss climbs/diverges instead of falling). The course calls
this the #1 beginner bug.

## Q2 — Context manager for the hand-written update

`with torch.no_grad():`

It is required because the update `w -= lr * w.grad` is itself a tensor operation on a tensor
with `requires_grad=True`. Without `no_grad()`, autograd would record the update step into the
computation graph as if it were part of the model's math (and complains about in-place
modification of a leaf tensor that requires grad). Wrapping it in `torch.no_grad()` tells
autograd "this bookkeeping operation is not part of the differentiable computation — don't track
it."

## Q3 — Broadcasting

Rule: right-align the shapes, pad the shorter one with 1s on the **left**; each dimension pair
must be equal or contain a 1; the output takes the larger of each pair.

- **(a)** `(2, 1, 4)` with `(5, 4)`: pad `(5, 4)` → `(1, 5, 4)`. Compare: `2 vs 1` → 2,
  `1 vs 5` → 5, `4 vs 4` → 4. **Result: `(2, 5, 4)`.**
- **(b)** `(3,)` with `(4,)`: single dimension, `3 vs 4` — neither is equal and neither is 1.
  **Error** — not broadcastable.
- **(c)** `img.mean(dim=(1, 2), keepdim=True)` reduces over H and W but `keepdim=True` keeps
  those axes as size 1, so the mean has shape **`(C, 1, 1)`**. The subtraction
  `img - mean` works because `(C, H, W)` vs `(C, 1, 1)` broadcasts: `C vs C` match, and the two
  size-1 dims stretch across H and W — each channel's scalar mean is subtracted from every pixel
  of that channel. (Without `keepdim` the mean would be `(C,)`, which right-aligns against `W`
  and either errors or broadcasts the wrong way.)

## Q4 — `nn.Linear` weight shape and the one-liner

`nn.Linear(in_features, out_features)` stores its weight as **`W.shape == (out_features,
in_features)`** — that's PyTorch's `(out, in)` convention.

Applying it to a batch `x` of shape `(N, in)`:

```python
y = x @ W.T + b
```

`W.T` is `(in, out)`, so `(N, in) @ (in, out) → (N, out)`; the bias `b` of shape `(out,)`
broadcasts across all N rows. **Output shape: `(N, out_features)`.**

## Q5 — `nn.CrossEntropyLoss` requirements

The two requirements:

1. The model output must be **raw logits** — unnormalized scores straight from the last
   `nn.Linear`, one per class, shape `(N, num_classes)`.
2. The target must be **integer class labels** (a `long` tensor of shape `(N,)`), e.g. `y = 2`,
   not a probability vector.

The two preprocessing steps beginners wrongly add:

1. Applying **softmax** to the logits before passing them in (CrossEntropyLoss does its own
   log-softmax internally; a second softmax is wrong).
2. **One-hot encoding** the labels (it wants class indices, not one-hot vectors).

## Q6 — Numerical gradient checking

Use the **central-difference** formula: for each parameter, perturb both ways and take

```
grad_i ≈ (f(x + eps·e_i) - f(x - eps·e_i)) / (2·eps)
```

(with `eps` ~ 1e-4), rather than a one-sided difference — central differences cancel the leading
error term and are far more accurate. Do it in **`float64`**: the computation subtracts two
nearly-equal numbers and divides by a tiny `eps`, so float32 round-off would swamp the answer.

Why it's far too slow for real training: it needs **two full forward passes per parameter per
step** (wiggle up, wiggle down). A model with a million parameters would need ~2 million forward
passes to get *one* gradient, every step. Backprop/autograd gets the entire gradient for the cost
of roughly one extra backward pass, by reusing intermediate results via the chain rule. Numerical
gradients are for *verifying* autograd (the finite-difference check), not for training.

## Q7 — CNN shape walk

Input: `(N, 1, 8, 8)`

| Stage | Output shape | Why |
|---|---|---|
| `nn.Conv2d(1, 8, kernel_size=3, padding=1)` | `(N, 8, 8, 8)` | `padding=1` with a 3×3 kernel keeps 8×8; channels 1 → 8 |
| `ReLU` | `(N, 8, 8, 8)` | elementwise, shape unchanged |
| `nn.MaxPool2d(2)` | `(N, 8, 4, 4)` | halves each spatial dim |
| flatten (keep batch dim) | `(N, 128)` | 8 channels × 4 × 4 = 128 features |
| `nn.Linear(128, 2)` | `(N, 2)` | so **`? = 128`** |

The #1 CNN bug this arithmetic prevents: a **size mismatch at the final `Linear`** — computing
the wrong flattened feature count so the fully-connected layer's `in_features` doesn't match
what the conv stack actually produces.

The call that flattens everything except the batch dimension: **`x.flatten(start_dim=1)`**.

## Q8 — Why a CNN beats an MLP on images

1. **Weight sharing / translation invariance.** A convolution slides the *same* small learnable
   filter across every position, so a pattern (edge, corner) is detected no matter where it
   appears — and the parameter count is tiny compared with a dense layer over all pixels.
2. **Locality / preserved spatial layout.** An MLP flattens the image and throws away the 2-D
   spatial structure; a CNN's filters look at local neighborhoods, so nearby pixels are treated
   as related, which is exactly the structure images have. Stacked layers then build up from
   edges → shapes → objects.

## Q9 — Dataset / DataLoader

(a) A custom `Dataset` must implement exactly **`__len__`** (how many samples) and
**`__getitem__(i)`**, which returns sample *i* as a **`(features, label)`** tuple.

(b) With 10 samples and `batch_size=4`, the `DataLoader` yields batches of **`[4, 4, 2]`** — the
last batch is the ragged leftover.

(c) Shuffle each epoch to avoid **order bias**: if the data is sorted (e.g. all of class 0, then
all of class 1), the model sees long runs of correlated samples and each gradient step is biased
toward whatever slice it's currently in. Shuffling makes each batch a representative sample so
the gradient estimates are unbiased and the model doesn't learn the ordering.

## Q10 — Optimizers and schedules

(a) **Adam vs AdamW:** they differ only in *how weight decay is applied* — AdamW **decouples**
the weight decay from the gradient/adaptive step (it's applied directly to the weights rather
than folded into the gradient like an L2 penalty). Constructor arguments are the same; the
difference is the class you instantiate.

(b) Linear warmup:

```
lr(t) = base * min(1.0, (t + 1) / warmup_steps)
```

The `+1` prevents step 0 from being a dead zero. Warmup solves the problem at the very start of
training: the weights are random, so the first gradients are **large and noisy** — a full-size
learning rate on them can blow the model up. Ramping the LR up gently lets the parameters settle
before taking full-size steps.

(c) Read the current LR off a live optimizer with:

```python
optimizer.param_groups[0]["lr"]
```

(a scheduler mutates this value in place on `scheduler.step()`).

## Q11 — Checkpointing

(a) A `state_dict` is an **ordered dictionary mapping every parameter and buffer name**
(`"fc1.weight"`, `"fc1.bias"`, …) **to its tensor**. The optimizer has one too (momentum
buffers, Adam running averages, step counts, learning rate).

(b) Save the `state_dict`, not the whole model object, because `torch.save(model, path)`
pickles the object, which **hard-codes your class's import path** — the checkpoint breaks the
moment you refactor or move the class. A `state_dict` is just named tensors; you can always load
it into a freshly constructed model. This is the #1 way checkpoints rot.

(c) A resume needs the **optimizer state** because optimizers carry state of their own — SGD
momentum buffers, Adam's running averages of gradients, step counts, the current LR. Reloading
only the weights resets those to zero, so the very next step is jolted (it behaves like step 0 of
a fresh run, not step N+1 of the old one).

(d) `model.load_state_dict(ckpt["model"])` mutates the model **in place** and returns a **report
of missing/unexpected keys** — it does *not* return a new model.

## Q12 — Transfer learning

(a) "Freezing" a parameter means setting **`p.requires_grad = False`**. Autograd then computes
**no gradient** for it — its `.grad` stays empty — so even an optimizer built over *all*
`model.parameters()` has nothing to apply and the parameter stays put. You freeze a whole module
by looping over `module.parameters()`.

(b) **Yes.** A freshly constructed `nn.Linear` has parameters with `requires_grad=True` by
default, so a newly assigned head trains automatically. (Caveat from the course: if you built the
optimizer before freezing and later *unfreeze*, make sure those params are actually in the
optimizer.)

(c) openpilot's driving model **is** this pattern: a **shared backbone with multiple task heads**
(path, lead car, lane lines). When the team adds a new prediction, they attach a new head to the
shared backbone rather than retrain the whole network — exactly the freeze/swap-head/fine-tune
workflow.

## Q13 — Why export, tracing's blind spot, `eval()`

(a) A trained `nn.Module` is **Python objects plus Python code** — it needs your class
definitions, your training script, and a full Python + PyTorch interpreter stack. A deployment
target (C++ service, phone, comma device) has none of that. Exporting captures the *computation*
in a portable, self-contained format that runs without your source code or Python.

(b) `torch.jit.trace` runs the model **once** on the example input and **records the ops that
actually executed** for that run. That means **data-dependent control flow** — e.g.
`if x.sum() > 0: ...` — is silently frozen: only the branch taken during tracing is recorded, and
the exported model will always run that branch, whatever the future input. `torch.jit.script`
and `torch.export` handle this by capturing the program/graph rather than one execution.

(c) `model.eval()` before exporting so that **inference-time behavior is baked in**: layers like
dropout and batchnorm behave differently in train vs eval mode (dropout must be off, batchnorm
must use its running statistics). Trace in train mode and you'd ship training behavior.

## Q14 — What TorchScript loading needs

To load and run `m.pt` elsewhere you need only **`torch.jit.load("m.pt")`** and a PyTorch
runtime. You do **not** need the original model class definition, your training script, or any
of your Python source — the traced module is self-contained.

The single check that proves fidelity: run the original and the exported model on the **same
fresh inputs** and confirm **`torch.allclose(original(x), loaded(x))`**. That
outputs-match check is the one every deployment pipeline runs.

## Q15 — ExecuTorch

(a) ExecuTorch is PyTorch's **tiny, portable C++ runtime** for executing exported models
**on-device with no Python at all** at inference time.

(b) The three-stage pipeline:

1. **Export** — `torch.export` the trained model to a clean, backend-agnostic graph
   (an `ExportedProgram`).
2. **Lower/compile** — compile that graph to a **`.pte`** file (optionally quantizing and
   assigning parts to hardware backends).
3. **Run** — execute the `.pte` on-device with the ExecuTorch C++ runtime.

(c) A **delegate** is a mechanism for offloading pieces of the graph to specialized hardware
accelerators (NPU/GPU/DSP) instead of running them on the generic runtime — the compiled graph
hands those subgraphs to the backend that runs them fastest.

(d) A fixed, ahead-of-time graph gives a real-time device what eager mode can't: **no Python
interpreter overhead, ahead-of-time optimization of the whole graph, and fixed, pre-plannable
memory**. For a device processing every camera frame on a deadline, predictable latency and a
known memory footprint are exactly what's needed.

## Q16 — `torch.export.export`

```python
model.eval()
exported = torch.export.export(model, (x,))   # note: a TUPLE of positional args
```

The second argument must be a **tuple** of the model's positional arguments — `(x,)`, not `x`.
You get back an `ExportedProgram`; run it via `exported.module()(x)`.

"Specializes on the example's shapes" means the exported graph is recorded for the **specific
shapes** of the example input. Afterward you must run it with inputs of **matching shapes** — an
input of a different shape won't work with that exported graph.

## Q17 — Quantization

(a) Int8 quantization stores the weights (and sometimes activations) as **8-bit integers instead
of 32-bit floats**. The model gets roughly **4× smaller** — because 32 bits / 8 bits = 4 —
in exchange for a small, usually acceptable, precision (accuracy) loss, plus faster compute.

(b) One-line dynamic quantization for the Linear layers:

```python
qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

(do it with the model in `eval()` mode; it returns a quantized copy).

(c) The three flavors: **dynamic quantization**, **static quantization**, and
**quantization-aware training (QAT)**. Dynamic is the easiest to apply; **QAT recovers the most
accuracy** (the model learns during training to be robust to the quantized precision).

(d) The two things to measure afterward:

1. **The size/speed win** — measure the model's footprint (e.g. serialize the `state_dict` to a
   `BytesIO` and read its length) and its inference latency: did it actually shrink ~4× / get
   faster?
2. **The accuracy/fidelity cost** — compare quantized outputs against the float model (small
   relative error, e.g. the grader's ~10% bound) and check task accuracy is still acceptable.

If you don't measure both, you can't say whether the trade was worth it.

## Q18 — Design: steering-angle regression, end to end

**Dataset & split.**
- Wrap the 100k (frame, steering_angle) pairs in a `Dataset` (`__len__` + `__getitem__(i)` →
  `(image_tensor, angle)`), or a `TensorDataset` if they fit in memory. Feed with a `DataLoader`
  (`batch_size` tuned to GPU memory, `shuffle=True` for training to avoid order bias — dashcam
  logs are time-ordered, which is exactly the sorted-data bias shuffling exists to break;
  `num_workers>0` to keep the model fed).
- **Hold out a validation split** with `random_split(ds, [n_train, n_val], generator=g)` (e.g.
  80/20) using a seeded local `torch.Generator` for reproducibility. Why: you must never score
  yourself on data you trained on — training accuracy tells you about memorization, not
  generalization.
- **Preprocess:** per-channel normalize each frame (`mean/std over dim=(1,2), keepdim=True`) so
  inputs are standardized — gradient descent behaves badly when features are on wildly
  different scales.

**Model family.**
- A **CNN**: stacked `Conv2d → ReLU → MaxPool2d` blocks, then `flatten(start_dim=1)`, then
  `Linear` layers ending in a **single output unit** (the predicted angle). Why a CNN: weight
  sharing and locality make it the right structure for images (Q8) — this is the same family as
  openpilot's vision model. Even better: **transfer learning** — take a pretrained vision
  backbone, freeze it, and replace the head with a 1-output regression head; the early layers'
  general features (edges, road textures) transfer, so you need less data and less compute.

**Loss function.**
- **MSE (mean squared error)** — `nn.MSELoss()` / mean of squared errors. Why: steering angle is
  a **single continuous value**, so this is *regression*, not classification. Softmax +
  cross-entropy make no sense with no classes; MSE directly penalizes the squared distance
  between predicted and human angle, exactly the Day 33 regression setup.

**Training loop.**
- `device = pick_device()`; move model and every batch onto it (`move_to(batch, device)`).
  Optimizer: `torch.optim.Adam(model.parameters(), lr=...)` (possibly AdamW + weight decay +
  linear warmup on a run this size). On a CUDA GPU, wrap the forward in
  `torch.autocast(device_type=device.type)` and the backward in a `torch.amp.GradScaler('cuda')`
  (`scaler.scale(loss).backward(); scaler.step(opt); scaler.update()`) — AMP roughly halves time
  and memory; the scaler prevents half-precision gradient underflow.
- The five steps per batch: `opt.zero_grad()` → forward → MSE loss → `loss.backward()` →
  `opt.step()`.
- **Monitor: validation loss every epoch** (`model.train()` for the training pass,
  `model.eval()` + `torch.no_grad()` for scoring). **Keep the best-validation checkpoint** via
  `copy.deepcopy(model.state_dict())` — the last epoch may be worse than an earlier one
  (overfitting). Also checkpoint model + optimizer + epoch to disk regularly
  (`torch.save({"model": ..., "optimizer": ..., "epoch": ...}, path)`) so a crash doesn't lose
  the run.

**Export & on-device lowering.**
- Load the best `state_dict` into a fresh model, call **`model.eval()`** (bake in inference
  behavior), then **`exported = torch.export.export(model, (example_frame,))`** — note the
  tuple. Lower/compile the exported graph to a **`.pte`** and run it with the **ExecuTorch C++
  runtime** on the in-car computer — no Python on the device, ahead-of-time optimized graph,
  fixed memory, optionally with **delegates** offloading conv-heavy subgraphs to the car
  computer's accelerator. This is the same train → export → lower path openpilot's driving
  model takes.

**Quantization.**
- Quantize to **int8** (e.g. `torch.ao.quantization.quantize_dynamic(model, {nn.Linear},
  dtype=torch.qint8)`, or static/QAT as part of the ExecuTorch lowering — QAT if the accuracy
  hit from post-training quantization is too big). Why: ~4× smaller and faster on constrained,
  battery/thermally-limited real-time hardware, for a small accuracy cost.

**The two verifications before trusting the deployed artifact:**
1. **Fidelity** — run the deployed (exported + quantized) model and the original float model on
   the **same fresh held-out frames** and confirm the outputs match: `torch.allclose` for the
   export step, and a small relative error bound (plus validation-set MSE re-measured on the
   quantized model) for the quantized version. The exported model must compute the same thing.
2. **Speed** — measure **on-device inference latency** and confirm it meets the real-time
   deadline (it must process every camera frame at the camera's rate, e.g. comfortably under
   the frame period, with predictable worst-case time). A faithful model that misses its
   deadline is useless in a car.

## Q19 — `train` skeleton with best-state keeping

```python
import copy

def train(model, train_loader, val_loader, E, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ...  # e.g. nn.CrossEntropyLoss() or nn.MSELoss()

    best_val = -1.0                                  # or +inf if tracking val loss
    best_state = copy.deepcopy(model.state_dict())   # the line that makes it SAFE

    for epoch in range(E):
        model.train()                                # mode toggle 1
        for X, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()

        model.eval()                                 # mode toggle 2
        with torch.no_grad():
            val_score = evaluate(model, val_loader)  # accuracy (or -val loss)

        if val_score > best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())

    return best_state
```

The two mode toggles are **`model.train()`** before the training pass and **`model.eval()`**
before validation (with `torch.no_grad()` around scoring). The one line that makes "keeping the
best state" actually safe is **`copy.deepcopy(model.state_dict())`** — a plain reference to the
`state_dict` keeps pointing at the *live* weights, so the next epoch would silently overwrite
your "best."

## Q20 — Validation discipline

(a) If validation overlaps training, you are scoring the model on data it has already seen — a
great score on training data tells you only that the model can **memorize**, not that it
**generalizes** to new inputs. A 100%-on-training-data model tells you essentially nothing about
how it will do on the next frame it has never seen; that number "always looks great and means
nothing."

(b) Keep the best-*validation* epoch because of **overfitting**: past some point the model keeps
improving on the training set while getting *worse* on held-out data. The last epoch can
therefore be worse than an earlier one; the best-val checkpoint is the one that actually
generalized best.

(c) The #1 bug the course warns about in real training loops: **scoring on the training set and
calling it "accuracy"** — always report validation metrics. (Its loop-mechanics sibling, the #1
beginner bug from Day 22: forgetting `zero_grad()`, so gradients accumulate across steps.)

## Q21 — Safety & contribution workflow

(a) All openpilot development and testing happens **on your own computer, against
recorded/simulated data** — replayed logged routes, no car or comma device required. You must
**never test unreviewed code on a real vehicle**. openpilot is real driving software; safety
comes first.

(b) A first PR should **avoid safety-critical vehicle-control logic** entirely, and should avoid
big, sprawling, multi-purpose changes. Good first contributions: **tooling/dev-experience
improvements, docs/typo fixes, tests, improved error messages, and small well-scoped bug fixes
backed by a test** — small and correct beats big and impressive; reviewers merge what they can
fully understand. Follow `CONTRIBUTING.md` to the letter.

(c) The new test must **fail before your fix and pass after it**. That property proves the test
actually pins the buggy behavior — reproduce first, watch the test fail, then make the minimal
change that turns it green (and breaks nothing else).
