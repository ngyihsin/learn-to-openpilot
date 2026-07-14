# Final Exam Answers — Student S2

---

## Q1 — The canonical training loop (5 pts)

The five steps, in order:

1. **Forward** — run the model on the batch: `pred = model(X)`
2. **Loss** — score how wrong it is: `loss = criterion(pred, y)`
3. **Backward** — compute gradients: `loss.backward()`
4. **Update (step)** — apply the update: `opt.step()` (or `w -= lr * w.grad` by hand)
5. **Zero grad** — clear the gradients: `opt.zero_grad()` (equivalently you can zero at the
   top of the loop before the forward pass — the point is once per step).

(a) Gradients must be zeroed every step because `loss.backward()` does not *overwrite*
`.grad` — it **accumulates** (adds) into it. `.grad` is a running sum, not a fresh value.

(b) If you forget, each step's gradient gets added on top of all the previous steps'
gradients, so the update direction becomes the sum of stale gradients from every epoch so
far. The effective step size grows and points in a direction that no longer matches the
current loss surface, so training goes haywire / diverges instead of descending. The course
called forgetting `zero_grad` the #1 beginner bug (Day 22/23).

---

## Q2 — Context manager for the manual update (4 pts)

`with torch.no_grad():`

The update `w -= lr * w.grad` is itself a tensor operation on a tensor that has
`requires_grad=True`. Without `no_grad()`, autograd would record the update step as part of
the computation graph — but the parameter update is bookkeeping, not part of the model's
computation, and we don't want gradients flowing through it (it would also error/behave
badly to do an in-place op on a leaf tensor that's being tracked). Inside `no_grad()` the
operation is not recorded, so the update is a plain numeric change.

---

## Q3 — Broadcasting (4 pts)

The rule (Day 23b): right-align the shapes, pad the shorter one with 1s on the **left**,
then compare dimension by dimension — each pair must be equal or contain a 1; output takes
the bigger.

- **(a) `(2, 1, 4)` with `(5, 4)`** → pad to `(2, 1, 4)` vs `(1, 5, 4)`. Columns:
  2 vs 1 ✓ → 2; 1 vs 5 ✓ → 5; 4 vs 4 ✓ → 4. **Result: `(2, 5, 4)`.**
- **(b) `(3,)` with `(4,)`** → **error**. Right-aligned, the only column is 3 vs 4: not
  equal and neither is 1, so they are not broadcastable.
- **(c)** With `dim=(1, 2), keepdim=True` the reduced axes are kept as size 1, so the mean
  has shape **`(C, 1, 1)`**. The subtraction `img - mean` works because `(C, H, W)` vs
  `(C, 1, 1)` broadcasts: C matches C, and the two 1s stretch across H and W — each
  channel's scalar mean is subtracted from that whole channel. (Without `keepdim` the mean
  would be `(C,)`, which right-aligns against W, not against the channel axis — wrong or
  an error.)

---

## Q4 — `nn.Linear` weight shape (4 pts)

`nn.Linear(in_features, out_features)` stores `W` with shape **`(out_features, in_features)`**
— that's PyTorch's convention, `(out, in)`.

One-line application to a batch `x` of shape `(N, in)`:

```python
y = x @ W.T + b
```

`x` is `(N, in)`, `W.T` is `(in, out)`, so the matmul gives `(N, out)`; the bias `b` of
shape `(out,)` broadcasts across all N rows. **Output shape: `(N, out_features)`.**

---

## Q5 — `nn.CrossEntropyLoss` requirements (4 pts)

Two requirements:

1. The model output must be **raw logits** — unnormalized scores, one per class, no softmax
   applied.
2. The target must be **integer class labels** (a `long` tensor of class indices), not
   probabilities.

The two things beginners wrongly add:

1. Applying **softmax** to the model output before the loss (CrossEntropyLoss does the
   softmax/log internally).
2. **One-hot encoding** the labels (it wants plain integer indices).

---

## Q6 — Numerical gradient check (4 pts)

Use **central differences**: for each parameter, wiggle it both ways and measure the loss:

```
grad_i ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

with a small `eps` (e.g. 1e-4), and do the arithmetic in **float64** — the formula subtracts
two nearly-equal numbers, so single precision loses too many significant digits. Central
(two-sided) differences are more accurate than a one-sided difference for the same eps.

Why it's too slow for real training: it needs **two full forward passes per parameter, per
step**. A model with a million parameters would need ~two million forward passes just to get
one gradient, every training step. Backprop/autograd gets *all* the gradients from one
forward + one backward pass, which is why it exists. The numerical version is only for
*checking* that the analytic gradient is right (the finite-difference check the graders ran
in Days 33/35/22).

---

## Q7 — Shape walk through the CNN (5 pts)

Input: `(N, 1, 8, 8)`

1. `nn.Conv2d(1, 8, kernel_size=3, padding=1)` → **`(N, 8, 8, 8)`** — padding=1 with a 3×3
   kernel keeps the 8×8 spatial size; channels go 1 → 8.
2. `ReLU` → **`(N, 8, 8, 8)`** — elementwise, shape unchanged.
3. `nn.MaxPool2d(2)` → **`(N, 8, 4, 4)`** — halves each spatial dimension.
4. Flatten (keep batch dim) → **`(N, 128)`** — because 8 channels × 4 × 4 = 128.
5. `nn.Linear(128, 2)` → **`(N, 2)`** — so **`? = 128`**.

The #1 CNN bug this arithmetic prevents: a **size mismatch at the final `Linear`** — getting
the flattened feature count wrong because you didn't track how conv/pool changed the spatial
size and channels.

The call that flattens everything except the batch dimension:
**`x.flatten(start_dim=1)`**.

---

## Q8 — Why a CNN beats an MLP on images (3 pts)

1. **Weight sharing / translation invariance.** A conv slides the *same* small filter over
   every position, so a pattern (an edge, a bright line) is detected no matter where it
   appears — and it needs far fewer parameters than a dense layer connecting every pixel to
   every unit.
2. **It preserves spatial layout / exploits locality.** An MLP flattens the image and throws
   away which pixels are neighbors; a conv looks at local patches, so nearby-pixel structure
   (exactly what images are made of) is built into the architecture. Stacked convs then
   build up edges → shapes → objects.

---

## Q9 — Dataset / DataLoader (4 pts)

(a) A custom `Dataset` must implement exactly two methods:
- `__len__` — returns how many samples there are;
- `__getitem__(i)` — returns sample i, as a `(features, label)` tuple.

(b) With 10 samples and `batch_size=4` the DataLoader yields batches of **`[4, 4, 2]`** —
the last batch is the ragged leftover.

(c) Shuffle each epoch to avoid **order bias**: if the data comes sorted (e.g. all of class
0 first), the model sees long runs of similar samples and the gradient steps get biased by
that ordering; shuffling makes each batch a representative mix and each epoch a different
ordering.

---

## Q10 — Optimizers & LR schedules (4 pts)

(a) Adam vs AdamW: they differ only in **how weight decay is applied** — AdamW *decouples*
the weight decay from the gradient/adaptive step (it applies the pull-toward-zero directly
to the weights instead of mixing it into the gradient that Adam then rescales). Same
constructor arguments, different class.

(b) Linear warmup:

```
lr(t) = base * min(1.0, (t + 1) / warmup_steps)
```

The `+1` makes step 0 not a dead zero. Warmup solves the problem at the very start of
training: the weights are random and the gradients are large and noisy, so a full-size
learning rate can blow the run up immediately. Ramping the LR up gently lets things settle
before taking full steps.

(c) Read the current LR off a live optimizer:

```python
optimizer.param_groups[0]['lr']
```

(A scheduler mutates that value in place when you call `scheduler.step()`, so read it
*before* stepping if you want the LR used for the current step.)

---

## Q11 — Checkpointing (5 pts)

(a) A `state_dict` is an **ordered dictionary mapping every parameter and buffer name to its
tensor** — e.g. `"fc1.weight" -> tensor`, `"fc1.bias" -> tensor`. The optimizer has its own
state_dict too (momentum buffers, step counts, learning rate).

(b) Save the `state_dict`, not the whole model object, because `torch.save(model, path)`
pickles the object, which **hard-codes your class's import path** into the file. The moment
you rename or move the class, the checkpoint won't load — the course called this the #1 way
checkpoints rot. A state_dict is just named tensors and loads into any compatible model.

(c) A resume needs the **optimizer state** because optimizers carry their own running state
— SGD momentum buffers, Adam's running averages of gradients, the step count/LR. If you
reload only the weights, that state starts from zero and the very next update step is
jolted (it behaves like step 1 of a fresh run, not step N+1 of your run).

(d) `model.load_state_dict(ckpt["model"])` mutates the model **in place** and returns a
**report of missing/unexpected keys** — it does *not* return a new model.

---

## Q12 — Transfer learning (4 pts)

(a) Freezing a parameter means setting **`param.requires_grad = False`** (you freeze a whole
module by looping over `module.parameters()`). Autograd then computes **no gradient** for
that parameter, so when the optimizer steps, there is nothing to apply — the parameter stays
put even if it's registered in the optimizer.

(b) Yes — a freshly constructed `nn.Linear` has parameters with `requires_grad=True` by
default, so the new head trains automatically.

(c) openpilot's driving model **is** this pattern: a shared vision backbone with multiple
task **heads** (path, lead car, lane lines). When the team wants a new prediction, they
attach a new head to the existing backbone rather than retraining the whole network.

---

## Q13 — Why export; tracing's blind spot; eval() (5 pts)

(a) A trained `nn.Module` is Python objects plus Python code — running it requires your
training script, your class definitions, and a full Python+PyTorch stack. A phone or the
comma device can't spin that up per camera frame. Exporting captures the *computation
itself* in a portable format that runs without your Python code.

(b) `torch.jit.trace` runs the model **once** on the example input and **records the
operations that actually executed**. That means **data-dependent control flow** — e.g.
`if x.sum() > 0: ... else: ...` — is silently frozen to whichever branch the example input
took. The trace has no `if` in it; other inputs that should take the other branch get the
wrong computation with no error. `torch.jit.script` / `torch.export` capture the control
flow properly.

(c) `model.eval()` before export bakes in inference-time behavior of layers that act
differently in training — dropout (must be off) and batchnorm (must use its running stats,
not batch stats). Otherwise you'd ship a model that still behaves like it's training.

---

## Q14 — Loading a traced model elsewhere (4 pts)

To load and run `m.pt` you need only **`torch.jit.load("m.pt")`** and a PyTorch runtime —
you do **not** need the original model class definition, your training script, or any of
your Python source. That's the whole point of the export.

The check that proves fidelity: **run both the original and the loaded/exported model on the
same fresh inputs and verify the outputs match with `torch.allclose`**. Every deployment
pipeline runs this check.

---

## Q15 — ExecuTorch (6 pts)

(a) ExecuTorch is PyTorch's **tiny, portable C++ runtime** for running exported models
on-device — at inference time there is **no Python at all**.

(b) The three-stage pipeline:
1. **Export** — `torch.export` the trained model into a clean, backend-agnostic graph
   (an `ExportedProgram`).
2. **Lower / compile** — compile that graph into a **`.pte`** file (optionally with parts
   assigned to hardware backends).
3. **Run** — execute the `.pte` on-device with the ExecuTorch C++ runtime.

(c) A **delegate** is a mechanism for offloading pieces of the graph to specialized hardware
accelerators (NPU/GPU/DSP-type backends) instead of running them on the general CPU runtime.

(d) A fixed, ahead-of-time graph gives the runtime what eager mode can't: no Python
interpreter overhead, the whole computation is known up front so it can be **optimized ahead
of time**, and memory can be planned/allocated as a **fixed** layout — all of which matter
for a battery-powered device that must hit a real-time deadline every camera frame.

---

## Q16 — `torch.export.export` (4 pts)

```python
model.eval()
exported = torch.export.export(model, (x,))   # note: the second arg is a TUPLE
runnable = exported.module()
y = runnable(x)
```

The second argument is a **tuple of the model's positional args** — even for a single
tensor you pass `(x,)`, not `x`.

"Specializes on the example's shapes" means the exported graph is recorded for the example
input's shapes; afterward you must **run it with inputs of matching shapes**. Feed it a
different shape and it won't work — the graph was fixed for the shapes it saw at export
time.

---

## Q17 — Quantization (6 pts)

(a) Int8 quantization stores the weights (and sometimes activations) as **8-bit integers**
instead of 32-bit floats. The model gets roughly **4× smaller** — because 8 bits is one
quarter of 32 bits. You also get faster compute, at the cost of a small precision hit.

(b) Dynamic quantization of the Linear layers (quantize in `eval()` mode):

```python
qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

(c) The three flavors: **dynamic** quantization, **static** quantization, and
**quantization-aware training (QAT)**. Dynamic is the easiest; **QAT recovers the most
accuracy** (the model is trained knowing it will be quantized).

(d) The two things to measure afterward:
1. **Size/speed** — did the model actually shrink (~4×) / get faster (measure the footprint,
   e.g. serialize the state_dict and compare bytes)?
2. **Accuracy/fidelity** — do the quantized outputs stay close to the float model's outputs
   (small relative error, e.g. within ~10%)? If the accuracy hit is too big, the size win
   wasn't worth it.

---

## Q18 — Design: steering-angle model, end to end (9 pts)

**1. Dataset & split.**
- Wrap the 100k (frame, steering_angle) pairs in a `Dataset` (`__len__` + `__getitem__`
  returning `(image_tensor, angle)`), served by a `DataLoader` with shuffling each epoch
  (avoid order bias — logged drives are heavily time-ordered) and a reasonable batch size.
- Preprocess: per-channel normalize the images (`mean/std` with `keepdim` broadcasting,
  Day 23b) so inputs are standardized — gradient descent behaves badly on wildly scaled
  features.
- **Split** into train and validation with `random_split` (e.g. 80/20) using a seeded
  generator so the split is reproducible. Never score on data you trained on — that's the
  only honest measure of generalization.

**2. Model family.**
- A **CNN**: camera frames are images, and convolutions preserve spatial layout, share
  weights, and detect local patterns wherever they appear (Day 24). Something like a stack
  of `Conv2d → ReLU → MaxPool` blocks, flatten (`start_dim=1`), then `Linear` layers down to
  **a single output unit** — the predicted steering angle. This is literally the shape of
  openpilot's vision model (camera frame → driving path).
- (If good pretrained vision backbones are available, transfer learning — freeze the
  backbone, replace the head with a 1-output regression head, fine-tune — would be faster
  and need less data. Day 25d.)

**3. Loss function.**
- **MSE (mean squared error)** on the predicted vs. true angle. Steering angle is a single
  **continuous** value — this is *regression*, not classification (Day 33). Softmax +
  cross-entropy make no sense here: there are no classes, no logits-per-class, and no
  probability distribution to form. MSE directly penalizes how far the predicted angle is
  from the human's, quadratically, and it's the loss whose gradients we derived by hand.

**4. Training loop (what to monitor, what to keep).**
- The canonical five steps per batch: forward → MSE loss → backward → `opt.step()` →
  `opt.zero_grad()`. Adam (or AdamW with weight decay) as the optimizer, possibly a warmup +
  decay LR schedule (Day 25b).
- Each epoch: `model.train()` for the training pass, then `model.eval()` +
  `torch.no_grad()` for a **validation pass**; monitor **validation loss** (mean squared /
  absolute angle error on held-out frames).
- **Keep the best-validation checkpoint**, not the last epoch: save
  `copy.deepcopy(model.state_dict())` (plus optimizer state and epoch, via `torch.save`)
  whenever val improves, so overfitting later can't destroy the best weights and a crash
  can't lose the run (Days 25c/28b).

**5. Export & on-device lowering.**
- `model.eval()`, then export: `torch.export.export(model, (example_frame_batch,))` — the
  clean backend-agnostic graph — then lower/compile to a **`.pte`** file for the
  **ExecuTorch** C++ runtime, since the in-car computer runs no Python (Days 26–27). The
  graph specializes on the example's shape, which is fine: camera frames have a fixed shape.

**6. Quantization.**
- Quantize to **int8** (dynamic quantization as the easy first pass; QAT if the accuracy hit
  is too large) — ~4× smaller and faster, which matters enormously on an embedded,
  real-time, power-constrained device (Day 28).

**7. Two verifications before trusting the artifact.**
1. **Fidelity:** run the exported/quantized model and the original float model on the same
   fresh validation frames and check the outputs match (`torch.allclose` for the export;
   small relative error, e.g. within a few percent of the float outputs, for the quantized
   version). The deployed thing must compute the same function you trained.
2. **Speed:** measure inference time per frame *on the target device* and confirm it beats
   the real-time deadline (the camera's frame rate — the model must finish each frame before
   the next arrives, every time, like openpilot's fixed-rate processes).

And per the capstone: all of this is validated against **recorded/replayed data** — never
test unreviewed models on a real vehicle.

---

## Q19 — `train` skeleton with best-state keeping (6 pts)

```python
import copy

def train(model, train_loader, val_loader, E, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ...  # e.g. nn.CrossEntropyLoss() or MSE for regression

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(E):
        model.train()                      # mode toggle 1
        for X, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()

        model.eval()                       # mode toggle 2
        with torch.no_grad():
            val_score = evaluate(model, val_loader)   # val accuracy (or -val loss)

        if val_score > best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())   # <-- the safety line

    return best_state
```

- The two mode toggles: **`model.train()`** before the training pass and **`model.eval()`**
  before scoring (no-op for a plain MLP, but essential once dropout/batchnorm exist).
- The line that makes keeping the best state safe:
  **`copy.deepcopy(model.state_dict())`** — a plain reference to the state_dict keeps
  pointing at the *live* tensors, so the next epoch would silently overwrite your "best."

---

## Q20 — Validation honesty (5 pts)

(a) The validation set must never overlap the training set because a great score on
training data tells you almost **nothing** — it only proves the model can fit (or memorize)
data it has already seen. The whole point of a model is to work on *new* inputs; only
held-out data measures that (generalization). 100% on training data is compatible with a
model that's useless in the field.

(b) Keep the best-validation epoch because of **overfitting**: past some point, more
training keeps pushing training loss down while validation performance gets *worse* — the
model starts memorizing training quirks. The last epoch can therefore be worse than an
earlier one; the best-val checkpoint is the model that actually generalized best.

(c) The #1 bug the course warns about in real training loops: **scoring yourself on the
training set and calling it "accuracy."** That number always looks great and means nothing —
always report validation. (The #1 bug inside the loop mechanics itself is forgetting
`zero_grad`, per Day 22 — but for a *real* full loop, the course's Day 28b warning is
train-set self-grading.)

---

## Q21 — Safety & workflow (capstone) (5 pts)

(a) All openpilot development and testing happens **on your own computer against
recorded/simulated data** (replaying logged routes through the stack — no car or comma
device needed). You must **never test unreviewed code on a real vehicle.**

(b) A first PR should **avoid safety-critical vehicle-control logic** entirely. Good first
contributions: docs/typo fixes, tooling and dev-experience improvements, tests, better error
messages, and small, well-scoped bug fixes — one focused change, backed by a test, following
`CONTRIBUTING.md` to the letter.

(c) The new test must **fail before your fix and pass after it**. That's what proves the
test actually pins the buggy behavior and that your change is what fixed it — a test that
passes either way proves nothing.
