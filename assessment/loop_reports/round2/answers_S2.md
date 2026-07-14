# Final Exam Answers — Student S2 (Round 2)

Background: EE/CS grad, no ML before this course. Studied Week 5 days 31–35 first, then all of
Week 4, then the capstone READMEs, as the syllabus recommended.

---

## Section 1 — Tensors, autograd & the training loop

### Q1 — The canonical training loop (5 pts)

The five named steps, in order:

1. **Forward** — run the model on the batch: `pred = model(X)`
2. **Loss** — score the output: `loss = criterion(pred, y)`
3. **Backward** — `loss.backward()` computes gradients for every parameter
4. **Step (update)** — `optimizer.step()` (or by hand, `w -= lr * w.grad`)
5. **Zero grad** — `optimizer.zero_grad()` (or `w.grad.zero_()`)

(Some code puts `zero_grad()` first, before the forward pass — Day 23 writes it that way. It's
the same loop rotated; all that matters is gradients are zeroed before the *next* `backward()`.)

(a) You must zero every step because **`loss.backward()` accumulates into `.grad`** — it adds the
new gradient onto whatever is already stored there, it does not overwrite.

(b) If you forget, the mechanism that breaks is accumulation: each step's `.grad` becomes the
**sum of the current gradient plus all the stale gradients from previous steps**. Your update is
then taken in the direction of that ever-growing sum, not the current slope, so the effective
step size grows and points in stale directions — training "goes haywire" (the course calls this
the #1 beginner bug). It's not an exception or a crash; it's silently wrong updates.

### Q2 — Which context manager for the manual update (4 pts)

`with torch.no_grad():`

The update `w -= lr * w.grad` is itself a tensor operation on a tensor with
`requires_grad=True`. Without `no_grad()`, autograd would record the update step as part of the
computation graph (and complain about in-place modification of a tensor that needs gradients).
We wrap it in `torch.no_grad()` so the parameter update is a plain arithmetic operation that
autograd does not track — the update is bookkeeping, not part of the model's computation.

### Q3 — Broadcasting (4 pts)

Rule (Day 23b): align the shapes **from the right**, pad the shorter one with 1s on the left;
each column must be equal or contain a 1; output takes the bigger of the two.

**(a) `(2, 1, 4)` with `(5, 4)`** — pad to `(2, 1, 4)` vs `(1, 5, 4)`:
- rightmost: 4 vs 4 ✓ → 4
- middle: 1 vs 5 ✓ (one is 1) → 5
- left: 2 vs 1 ✓ → 2

Result: **`(2, 5, 4)`**.

**(b) `(3,)` with `(4,)`** — **error**. Right-aligned, the only column is 3 vs 4: they're not
equal and neither is 1, so broadcasting fails.

**(c)** `img.mean(dim=(1, 2), keepdim=True)` on a `(C, H, W)` image reduces over H and W but
`keepdim=True` keeps those axes as size 1, so the mean has shape **`(C, 1, 1)`**. The
subtraction `img - mean` works because `(C, H, W)` vs `(C, 1, 1)` broadcasts: right-aligned,
W vs 1 ✓, H vs 1 ✓, C vs C ✓ — the per-channel mean stretches across every pixel of its
channel. (Without `keepdim` the mean would be shape `(C,)`, which right-aligns against W and
either errors or broadcasts the wrong way.)

### Q4 — `nn.Linear` weight shape (4 pts)

PyTorch's `nn.Linear(in_features, out_features)` stores its weight as
**`W` of shape `(out_features, in_features)`** — that's the `(out, in)` convention.

Applying it to a batch `x` of shape `(N, in)`:

```python
y = x @ W.T + b
```

Shapes: `(N, in) @ (in, out) → (N, out)`, and the bias `b` of shape `(out,)` broadcasts across
all N rows. Output shape: **`(N, out_features)`**.

### Q5 — `nn.CrossEntropyLoss` requirements (4 pts)

The two requirements:

1. The model output must be **raw logits** — unnormalized scores, one per class (no softmax
   applied).
2. The target must be **integer class labels** (a `long` tensor of class indices), not vectors.

The two preprocessing steps beginners wrongly add:

1. Applying **softmax** to the model output before the loss (CrossEntropyLoss does that
   internally, in a numerically stable way).
2. **One-hot encoding** the labels (it wants plain integer indices).

### Q6 — Numerical gradient checking (4 pts)

Use **central differences** — wiggle the parameter both ways and take the symmetric slope:

```
grad ≈ (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

not a one-sided difference (central differences are more accurate). Do it in **`float64`** so
the tiny differences aren't swamped by floating-point rounding error at `float32` precision.
This is the "finite-difference check" the graders used to prove hand-derived gradients match
reality.

Why it's too slow to train with: it costs **two forward passes per parameter** to get one
parameter's gradient. A million-parameter model would need millions of forward passes to
compute a single update. One `backward()` call gets **every** gradient at once for roughly the
cost of another forward pass — that is the entire reason autograd/backprop exists. Numerical
gradients are for *checking*, never for training.

---

## Section 2 — Building & feeding models

### Q7 — Shape walk through a small CNN (5 pts)

Input: `(N, 1, 8, 8)`

1. `nn.Conv2d(1, 8, kernel_size=3, padding=1)` → **`(N, 8, 8, 8)`** — padding=1 with a 3×3
   kernel keeps the spatial size 8×8; channels go 1 → 8.
2. `ReLU` → **`(N, 8, 8, 8)`** — elementwise, shape unchanged.
3. `nn.MaxPool2d(2)` → **`(N, 8, 4, 4)`** — halves each spatial dimension.
4. Flatten (keep batch dim) → **`(N, 128)`** — 8 channels × 4 × 4 = 128 features.
5. `nn.Linear(128, 2)` → **`(N, 2)`**. So **`? = 128`**.

The #1 CNN bug this arithmetic prevents: a **size mismatch at the final `Linear` layer** —
getting the flattened feature count wrong because you mis-traced how conv/pool changed the
spatial size, so the Linear's `in_features` doesn't match what actually arrives.

The call that flattens everything except the batch dimension: **`x.flatten(start_dim=1)`**.

### Q8 — Why a CNN beats an MLP on images (3 pts)

1. **Weight sharing / translation invariance.** A convolution slides the *same* small learnable
   filter across every position, so a pattern (an edge, a corner) is detected no matter where it
   appears — and the parameter count is tiny compared to a dense layer over all pixels.
2. **It preserves spatial layout / exploits locality.** An MLP flattens the image and throws
   away which pixels are neighbors; a CNN's filters look at local neighborhoods, building up
   from edges to shapes to objects. Spatial tasks ("is the bright line horizontal or
   vertical?") are exactly what that structure is built for.

### Q9 — Dataset / DataLoader (4 pts)

(a) A custom `Dataset` must implement exactly two methods:
- `__len__` — returns how many samples there are;
- `__getitem__(i)` — returns sample `i`, as a `(features, label)` tuple.

(b) 10 samples with `batch_size=4` → batches of **`[4, 4, 2]`** — the last batch is the ragged
leftover.

(c) Shuffle each epoch to **avoid order bias**: if the data arrives sorted (e.g., all of one
class first), the model sees systematically ordered batches and the gradient steps are biased
by that ordering. Shuffling makes each epoch's batches representative of the whole set.

### Q10 — Optimizers & LR scheduling (4 pts)

(a) **Adam vs AdamW:** the single difference is *how weight decay is applied* — **AdamW
decouples the weight decay from the gradient/adaptive step** (a direct pull toward smaller
weights), whereas Adam folds it in with the gradient (like an L2 penalty going through the
adaptive machinery). Same constructor arguments; different class.

(b) Linear warmup:

```
lr(t) = base * min(1.0, (t + 1) / warmup_steps)
```

The `+1` means step 0 isn't a dead zero. After `warmup_steps` it holds at `base`. Warmup solves
the problem at the very start of training: the weights are random, so the gradients are **large
and noisy** — taking full-size steps immediately can blow the run up. Ramping the LR up gently
lets the model settle before full-speed updates.

(c) Read the current LR off a live optimizer with:

```python
optimizer.param_groups[0]['lr']
```

(A scheduler mutates that value in place when you call `scheduler.step()`, so read it *before*
stepping if you want the LR that was actually used.)

### Q11 — Checkpointing (5 pts)

(a) A **`state_dict`** is an ordered dictionary mapping every parameter and buffer *name*
(`"fc1.weight"`, `"fc1.bias"`, …) to its tensor. The optimizer has one too (momentum buffers,
step counts, learning rate).

(b) Save the `state_dict`, not the whole model object, because `torch.save(model, path)`
pickles the object, which **hard-codes your class's import path** into the file — the moment
you rename or refactor the class/module, the checkpoint no longer loads. The course calls this
"the #1 way checkpoints rot." A `state_dict` is just named tensors: portable, loadable into any
fresh instance of the architecture.

(c) A *resume* checkpoint needs the optimizer state because optimizers like SGD-with-momentum
and Adam carry **internal running state** (momentum buffers, Adam's running averages, step
counts, the LR). If you restore only the weights, that state restarts from zero and the very
next update step is jolted — you are not resuming "exactly where you left off."

(d) `model.load_state_dict(ckpt["model"])` mutates the model **in place** and returns **a
report of missing/unexpected keys** — it does *not* return a new model.

### Q12 — Transfer learning (4 pts)

(a) "Freezing" a parameter means setting **`requires_grad = False`** on it. Autograd then
computes **no gradient** for that parameter, so after `backward()` there is nothing for the
optimizer to apply — it has no gradient to use, so the parameter stays put (even if it's in the
optimizer's parameter list).

(b) **Yes** — a freshly built `nn.Linear` has parameters with `requires_grad=True` by default,
so the new head trains automatically.

(c) openpilot's driving model is itself this pattern: a **shared backbone with multiple task
heads** (path, lead car, lane lines). When the team adds a new prediction, they **attach a new
head** rather than retrain the whole network — exactly the freeze-backbone / swap-head /
fine-tune workflow.

---

## Section 3 — Export, ExecuTorch, quantization

### Q13 — Why export; tracing's blind spot; eval() (5 pts)

(a) A trained `nn.Module` is **Python objects and Python code** — it needs your class
definition, your training script, and a full Python + PyTorch stack to run. A C++ service, a
phone, or the comma device can't (and shouldn't) spin all that up per camera frame. Exporting
captures the *computation itself* in a portable format that runs without your training code.

(b) `torch.jit.trace` **records the operations executed during one example run** with one
example input. Its blind spot is **data-dependent control flow** — e.g. `if x.sum() > 0: ...`.
Tracing only sees the branch that the example input happened to take; the recorded graph bakes
in that one path and silently computes the wrong thing for inputs that should take the other
branch. `torch.jit.script` / `torch.export` handle this by capturing the program logic rather
than one recorded run.

(c) Call `model.eval()` first so that **inference-time behavior is baked in** — layers like
dropout and batchnorm behave differently in train vs eval mode, and you want the exported
artifact to compute the deployment-time (eval) version, not the training-time version.

### Q14 — Loading a traced model elsewhere (4 pts)

To load and run `m.pt` elsewhere you need: **the file itself and `torch.jit.load("m.pt")`**
(i.e., a PyTorch/TorchScript runtime). What you **don't** need: the **original model class
definition** or any of your training code — the traced file is self-contained.

The single check that proves fidelity: **run both the original and the exported model on the
same fresh inputs and check `torch.allclose(original_out, exported_out)`** — the exported model
must compute the same thing as the original. Every deployment pipeline runs this check.

### Q15 — ExecuTorch (6 pts)

(a) ExecuTorch is PyTorch's **tiny, portable C++ runtime** for executing exported models
on-device — at inference time there is **no Python interpreter at all**.

(b) Three-stage pipeline:
1. **Export** the trained model with `torch.export` into a clean, backend-agnostic graph
   (an `ExportedProgram`);
2. **Lower/compile** that graph into a **`.pte` file**;
3. **Run** the `.pte` with the ExecuTorch C++ runtime on the device, optionally offloading
   pieces to hardware accelerators.

(c) A **delegate** is a mechanism by which specialized hardware claims part (or all) of the
exported graph: because the graph is explicit, a delegate can take a subgraph and hand it to a
DSP, NPU, or GPU while the C++ runtime executes the rest.

(d) A fixed, ahead-of-time graph helps a real-time device because with no Python interpreter in
the loop, the runtime can **optimize the whole graph before it ever runs** and **plan its
memory once** — which makes latency predictable. Predictable latency is exactly what a 20 Hz
real-time deadline demands; eager mode can't offer that.

### Q16 — `torch.export.export` (4 pts)

```python
model.eval()
exported = torch.export.export(model, (x,))
```

The second argument must be a **tuple** of the model's positional args — hence `(x,)` with the
trailing comma, even for a single input. You get back an `ExportedProgram`; call
`exported.module()` to get a runnable module.

"Specializes on the example's shapes" means the exported graph is fixed to the shapes of the
example input you exported with — so afterward you must **run it with inputs of matching
shapes**; feed a different shape and it won't work like the flexible eager model did.

### Q17 — Quantization (4 pts... actually 6)

(a) Int8 quantization stores the weights as **8-bit integers instead of 32-bit floats**. The
model shrinks roughly **4×**, because 32 bits / 8 bits = 4 — each weight takes a quarter of the
storage. The cost is a small precision/accuracy hit, usually acceptable.

(b) Dynamic quantization of the Linear layers (quantize in `eval()` mode; returns a quantized
copy):

```python
qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

(c) The three flavors, in increasing order of effort and accuracy recovered:
1. **Dynamic** — weights quantized ahead of time, activations on the fly (easiest);
2. **Static** — additionally pre-computes activation scales from a small calibration set;
3. **Quantization-aware training (QAT)** — simulates int8 *during* training so the model learns
   around the quantization error. **QAT recovers the most accuracy.**

Rule of thumb from the course: reach for the cheapest flavor that keeps your validation numbers.

(d) After quantizing, measure two things:
1. **Fidelity/accuracy** — that the quantized model's outputs stay close to the float model's
   (small relative error; validation metric holds up);
2. **The win itself — size and speed** — the model's footprint (serialize the `state_dict` and
   measure bytes; expect ~4× smaller) and its inference latency on the target. If accuracy
   dropped and you didn't actually get smaller/faster, it wasn't worth it.

---

## Section 4 — Applied: an autonomous-driving inference project

### Q18 — End-to-end design: steering-angle regression (9 pts)

**Dataset & split**
- Wrap the 100,000 (frame, steering_angle) pairs in a `Dataset` (`__len__` +
  `__getitem__(i)` → `(image_tensor, float_target)`). Targets are **floats** — no `long()`,
  no one-hot; this is regression.
- Preprocess: per-channel normalization of each frame (mean 0, std 1 per channel with
  `keepdim` broadcasting) so features are on comparable scales and gradient descent behaves.
- **Split into train and validation** (e.g. `random_split` with a seeded local generator for
  reproducibility). Why: you must never score yourself on data you trained on — a great
  training score only proves memorization, not generalization.
- Feed with a `DataLoader`: batching, shuffling each epoch (avoid order bias), parallel
  loading so the model isn't starved.

**Model family**
- A **CNN**: convolutions share filter weights across positions and preserve spatial layout,
  which is exactly right for camera frames (an MLP flattens away the spatial structure). Conv
  → ReLU → pool stack, then a flatten and a **head with ONE continuous output** — this is the
  same family as openpilot's vision model, which regresses driving quantities from frames.

**Loss function (not a class!)**
- **`nn.MSELoss`** — the target is a continuous number, so we penalize squared error, exactly
  the MSE built by hand on Day 33. CrossEntropy is *impossible* here, not just wrong: there is
  no finite set of classes; steering is a continuum.
- If the human-driver labels contain outliers (jerky corrections), prefer
  **`nn.SmoothL1Loss` (Huber)** — squared error explodes on outliers; Huber is the robust cousin.
- Watch the #1 regression shape bug: model outputs `(N, 1)`, targets `(N,)` — broadcasting
  silently makes an `(N, N)` difference matrix. `squeeze(-1)` the prediction before the loss.

**Training loop (monitor and keep)**
- Day 28b skeleton: per epoch — `model.train()`, loop batches: zero_grad → forward → loss →
  backward → step; then `model.eval()` and evaluate under `torch.no_grad()`.
- **Monitor validation MAE** (mean absolute error) — "accuracy" is meaningless for a
  continuous output; MAE says "on average, how many units off is the steering?"
- **Keep the best-val-MAE checkpoint**, saved as `copy.deepcopy(model.state_dict())` (plus
  optimizer state and epoch if I want to resume). Best epoch, not last — the last may be
  overfit. Checkpoint early and often so a crash doesn't lose the run.

**Export & on-device lowering**
- `model.eval()`, then export: `torch.export.export(model, (example_frame,))` (or
  `torch.jit.trace` for the TorchScript path) — the artifact must run without my Python
  training code.
- Lower/compile the exported graph to a **`.pte`** and run it with the **ExecuTorch** C++
  runtime on the embedded computer — no Python interpreter, AOT-optimized graph, memory
  planned once, optionally delegating subgraphs to an NPU/DSP.

**Quantization**
- Quantize to **int8** (start with dynamic quantization; move to static or QAT only if
  validation MAE degrades too much). ~4× smaller and faster — vital on a constrained,
  real-time, in-car computer.

**The two verifications before trusting the deployed artifact**
1. **Fidelity** — the deployed artifact computes what the trained model computes: run both on
   the same fresh inputs and check the outputs match (`torch.allclose` / small relative error
   after quantization), and confirm val MAE holds on the quantized+exported model.
2. **Speed** — measure inference latency **on the deployed artifact on the target device**
   (not the eager model on my laptop): a few warmup passes, then time many runs with
   `time.perf_counter()`, report the **median**. Check it against the frame budget: at 20 Hz
   the camera gives you `1000 / 20 = 50 ms` per frame; the model meets the budget only if its
   latency ≤ that. (In a car I'd also look at worst-case/tail latency, since a rare slow frame
   still matters.)

### Q19 — `train` function skeleton (6 pts)

```python
import copy

def train(model, train_loader, val_loader, optimizer, criterion, E):
    best_metric = float("inf")        # e.g. val loss / MAE; use -inf & > if accuracy
    best_state = None

    for epoch in range(E):
        model.train()                              # mode toggle #1
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()                               # mode toggle #2
        with torch.no_grad():
            val_metric = evaluate(model, val_loader)   # e.g. val MAE / loss

        if val_metric < best_metric:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())   # <-- the safety line

    return best_state
```

The two mode toggles are `model.train()` before the training pass and `model.eval()` before
scoring (plus `torch.no_grad()` around evaluation). The one line that makes "keeping the best
state" actually safe is **`copy.deepcopy(model.state_dict())`** — a plain reference keeps
pointing at the *live* weights, so the next epoch would silently overwrite your "best."

### Q20 — Validation honesty (5 pts)

(a) The validation set must never overlap the training set because a score on data the model
was trained on measures **memorization, not generalization**. A model can be great on its
training data and useless on anything new — "that number always looks great and means nothing."
The whole point of validation is to estimate performance on data the model has never seen.

(b) Keep the *best-validation* epoch instead of the last because of **overfitting**: past some
point the model keeps improving on the training set while getting *worse* on validation. The
last epoch can therefore be worse than an earlier one; the best-val checkpoint is the model
that actually generalized best.

(c) The #1 bug the course warns about: **forgetting to zero the gradients** each step —
`backward()` accumulates into `.grad`, so gradients pile up across steps and training goes
haywire (Day 22's #1 beginner bug). (The course also names the #1 *evaluation* bug: scoring on
the training set and calling it "accuracy.")

### Q21 — Safety & workflow (capstone) (5 pts)

(a) All openpilot development and testing happens **on your own computer against
recorded/simulated data** — replaying logged routes through the stack; no car or comma device is
needed to learn, contribute, and get PRs merged. What you must never do: **test unreviewed code
on a real vehicle**.

(b) A first PR should **avoid safety-critical vehicle-control logic** entirely. Good first
contributions: **tooling and dev-experience fixes, docs/typo fixes, tests, improved error
messages, and small, well-scoped bug fixes** — small and correct beats big and impressive, and
reviewers merge changes they can fully understand. Follow the project's `CONTRIBUTING.md` to
the letter.

(c) The new test must **fail before your fix and pass after it**. That property is what proves
the test actually pins the buggy behavior and that your change is what fixed it — a change with
such a test is far more likely to merge than a bare fix.
