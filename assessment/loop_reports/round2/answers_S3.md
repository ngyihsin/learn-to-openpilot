# Final Exam — Answers (Student S3, round 2)

I worked from the lesson READMEs (week 4, days 33–35 of week 5, and the capstone pages).
Where I'm not sure I say so.

## Q1 — the canonical training loop

The five steps, in order:

1. **forward** — run the model on the batch: `pred = model(X)`
2. **loss** — score how wrong it is: `loss = criterion(pred, y)`
3. **backward** — `loss.backward()` computes the gradients
4. **update** — `opt.step()` (or by hand, `w -= lr * w.grad`)
5. **zero_grad** — `opt.zero_grad()` so the gradients are clean for next time

(Day 23 writes the same loop "rotated" with `zero_grad` first — the lesson said that's fine,
all that matters is the grads are zeroed before the next `backward()`.)

(a) You must zero every step because `loss.backward()` does not overwrite `.grad` — it
**accumulates** into it. Each backward call adds its gradient on top of whatever is already
sitting in `.grad`.

(b) If you forget, the gradients from every previous step pile up in `.grad`, so your update
uses the sum of all past gradients instead of just the current one. The steps get bigger and
point in stale directions and training "goes haywire" — the course called this the #1
beginner bug. The mechanism is gradient accumulation across steps.

## Q2 — the context manager for the hand-written update

`with torch.no_grad():`

The update `w -= lr * w.grad` is itself a tensor operation on `w`. If you do it outside
`no_grad`, autograd records the update step as part of the computation graph, which you don't
want — the update is bookkeeping, not part of the model's math. Wrapping it in
`torch.no_grad()` means the weight update isn't tracked by autograd.

## Q3 — broadcasting

The rule from Day 23b: pad the shorter shape with 1s on the **left** until both are the same
length, then compare dimension by dimension; two dims are compatible if they're equal or
either is 1, and the output takes the bigger one.

(a) `(2, 1, 4)` with `(5, 4)` → pad the second to `(1, 5, 4)`. Compare: 2 vs 1 → 2; 1 vs 5 →
5; 4 vs 4 → 4. Result: **(2, 5, 4)**.

(b) `(3,)` with `(4,)` → same length already; compare 3 vs 4 — not equal and neither is 1, so
this is an **error** (that's the ValueError case).

(c) With `keepdim=True` the mean keeps its reduced dims as size 1, so the mean has shape
**(C, 1, 1)**. The subtraction `img - mean` works because `(C, 1, 1)` broadcasts cleanly back
over `(C, H, W)` — the 1s stretch to H and W. (Without keepdim the mean would just be `(C,)`
and it would line up against the wrong axis or error.)

## Q4 — nn.Linear weight shape

`nn.Linear` stores `W` as **(out_features, in_features)** — that's the PyTorch convention
from Day 23b. So you transpose it before the matmul:

```python
y = x @ W.T + b
```

Shapes: `(N, in) @ (in, out) → (N, out)`, and the bias `(out,)` broadcasts across all N rows.
Output shape: **(N, out_features)**.

## Q5 — CrossEntropyLoss requirements

The two requirements:
1. The model output must be **raw logits** — no softmax applied.
2. The target must be **integer class labels** (a long vector), not one-hot vectors.

The two things beginners wrongly add:
1. Applying **softmax** to the outputs before the loss.
2. **One-hot encoding** the labels.

The Day 28b hint literally said "no softmax, no one-hot."

## Q6 — checking a gradient numerically

Use **central differences** — wiggle the parameter both ways, not just one side (a one-sided
difference is less accurate). Use **float64** for accuracy, because the wiggles are tiny and
float32 rounding would swamp them. Roughly: `(loss(w + h) - loss(w - h)) / (2h)` per parameter.

Why it's too slow for real training: central differences cost **two forward passes per
parameter**. A million-parameter model would need millions of forward passes for a single
update. One call to `backward()` gets every gradient at once — that's the whole reason
autograd and backprop exist. So numerical gradients are for *checking*, never training.

## Q7 — CNN shape walk

Input: `(N, 1, 8, 8)`

- after `Conv2d(1, 8, kernel_size=3, padding=1)` → `(N, 8, 8, 8)` (padding=1 keeps 8×8, channels go 1→8)
- after `ReLU` → `(N, 8, 8, 8)` (no shape change)
- after `MaxPool2d(2)` → `(N, 8, 4, 4)` (halves the spatial size)
- after flatten (keeping batch) → `(N, 128)` because 8 channels × 4 × 4 = 128
- `nn.Linear(128, 2)` → `(N, 2)`

So `?` = **128**.

The #1 CNN bug this prevents is getting the **input size of the final Linear wrong** — a
mismatch between what the conv stack actually produces and what the Linear expects. The
lesson said "shapes are the whole battle."

The call that flattens everything except the batch dim: `x.flatten(start_dim=1)`.

## Q8 — why a CNN beats an MLP on images

1. An MLP flattens the pixels, which **throws away the spatial layout**. A CNN keeps the 2-D
   structure and looks at local neighborhoods, which is what image patterns are.
2. A CNN slides the **same small filter across every position**, so it detects a pattern (an
   edge, a lane line) no matter where in the image it appears — and reusing one filter
   everywhere means far fewer parameters than a dense layer connecting every pixel to every
   unit.

## Q9 — Dataset / DataLoader

(a) Exactly two methods: `__len__` (how many samples) and `__getitem__(i)`, which returns the
tuple `(features, label)` for sample i.

(b) 10 samples with `batch_size=4` → batches of **[4, 4, 2]** — the last one is the ragged
leftover.

(c) Shuffling each epoch avoids **order bias** — if the data arrives sorted (say all of one
class first), the model sees a biased stream and training gets skewed by the ordering instead
of the actual data distribution.

## Q10 — optimizers and warmup

(a) Adam vs AdamW: they take the same constructor arguments; the one difference is **how
weight decay is applied** — AdamW *decouples* the weight decay from the gradient step, plain
Adam mixes it into the gradient.

(b) Linear warmup:

```
lr(t) = base * min(1.0, (t + 1) / warmup_steps)
```

The `+1` is the off-by-one fix so step 0 isn't a dead zero. What it solves: at the very start
the weights are random, so the first gradients are big and unreliable — if you hit them with
the full learning rate immediately the run can blow up / diverge to NaN. Warmup ramps the lr
up gently so the early steps are small, then holds at `base`.

(c) Read it off the live optimizer with `optimizer.param_groups[0]['lr']` — and read it
*before* calling `scheduler.step()`, because the scheduler mutates that value in place.

## Q11 — checkpointing

(a) A `state_dict` is an ordered dictionary mapping every parameter and buffer *name*
(like `"fc1.weight"`, `"fc1.bias"`) to its tensor.

(b) Because pickling the whole model object hard-codes your class path — the moment you
refactor or move the class, the checkpoint won't load. The lesson called that "the #1 way
checkpoints rot." The state_dict is just named tensors, so it stays portable.

(c) The optimizer has its own state — momentum buffers, step counts, the learning rate
(Adam's running averages). If you resume with only the weights, all of that is reset and the
very next step gets jolted — you're not really resuming "exactly where you left off."

(d) `load_state_dict` mutates the model **in place** and returns a report of
missing/unexpected keys. It does **not** produce a new model — that trips people up.

## Q12 — transfer learning

(a) Freezing a parameter means setting `requires_grad = False` on it. Autograd then computes
**no gradient** for that parameter, so the optimizer has nothing to update — it stays put,
even if the optimizer was built over all the parameters.

(b) Yes — a freshly built `nn.Linear` has `requires_grad=True` on its params by default, so
the new head trains automatically.

(c) openpilot's driving model is itself a **shared backbone with multiple task heads** (path,
lead car, lane lines). When the team wants a new prediction they attach a new head rather than
retrain the whole network — exactly the freeze/swap-head/fine-tune pattern.

## Q13 — why export, and tracing's blind spot

(a) A trained `nn.Module` is Python objects and code — it needs your training script and a
Python interpreter to run. A C++ service or a device can't (and shouldn't) carry all that.
Exporting captures the *computation itself* in a portable format that runs without your
Python code.

(b) `torch.jit.trace` runs the model **once** on the example input and records the ops that
actually executed. So **data-dependent control flow** — something like `if x.sum() > 0:` —
gets silently baked in as whichever branch ran during the trace. Different data that should
take the other branch will still follow the recorded path. `script` / `torch.export` handle
this better.

(c) `model.eval()` before exporting bakes in the inference-time behavior of layers like
dropout and batchnorm — you don't want training-mode randomness frozen into the shipped
artifact.

## Q14 — loading the traced model

To load and run `m.pt` elsewhere you need `torch.jit.load("m.pt")` (and a torch runtime to
call it). What you do **not** need: the original class definition or your training script —
the lesson stressed "no original class definition required."

The single check that proves faithfulness: run both the original and the exported model on
the same fresh inputs and confirm the outputs match with `torch.allclose`. That's the check
every deployment pipeline runs.

## Q15 — ExecuTorch

(a) ExecuTorch is PyTorch's tiny, portable **C++ runtime** that executes exported models
on-device — with **no Python at all** at inference time.

(b) The three-stage pipeline: (1) **export** the trained model with `torch.export` to a
clean, backend-agnostic graph; (2) **compile/lower** that graph to a **`.pte` file**;
(3) **run** the `.pte` with the ExecuTorch runtime on the device.

(c) A **delegate** is a mechanism that claims part (or all) of the graph and hands that piece
to specialized hardware — a DSP, NPU, or GPU — while the C++ runtime executes the rest.

(d) Because the graph is fixed ahead of time, there's no Python interpreter in the loop, the
runtime can optimize the graph before it ever runs, and it can plan its memory **once**. That
is what makes latency predictable enough for a real-time device like the comma device that
has to answer every camera frame.

## Q16 — torch.export.export

```python
exported = torch.export.export(model, (x,))
```

The second argument must be a **tuple** of the model's positional args — even for one tensor,
it's `(x,)` with the comma. (Put the model in `eval()` first; `exported.module()` gives you
something you can call.)

"Specializes on the example's shapes" means the exported graph is recorded for the shape of
the example input you gave it — so afterwards you have to run it with inputs of **matching
shapes**, not arbitrary ones.

## Q17 — quantization

(a) It stores the weights as **8-bit integers instead of 32-bit floats**. The model gets
roughly **4× smaller** — because 8 bits is a quarter of 32 bits. You pay a small (usually
acceptable) precision/accuracy cost.

(b)

```python
qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

(done in `eval()` mode; it returns a quantized copy).

(c) The three flavors, in increasing order of effort and of accuracy recovered:
**dynamic** (weights ahead of time, activations on the fly — easiest), **static** (also
pre-computes activation scales from a small calibration set), and **quantization-aware
training (QAT)** — QAT recovers the most accuracy because the model learns around the int8
error during training. Rule from the lesson: reach for the cheapest one that keeps your
validation numbers.

(d) The two things to measure afterwards: (1) the **size/speed win** — did the model actually
shrink (~4×) and get faster; (2) the **accuracy/fidelity** — that the quantized outputs stay
close to the float model (the grader checked ~10% relative error; in practice, that your
validation numbers hold up).

## Q18 — design: steering-angle model from 100k dashcam frames

- **Dataset & split.** Wrap the frames + steering labels in a `Dataset`
  (`__len__`/`__getitem__` returning `(image_tensor, steering_float)`), feed it through a
  `DataLoader` with shuffling. Split into **train and validation** with `random_split` (with
  a seeded generator so it's reproducible). Why: you never score yourself on data you trained
  on — the training score always looks great and means nothing.
- **Preprocessing.** Standardize the inputs (per-channel mean/std normalization with
  keepdim broadcasting) — gradient descent behaves badly when features are on wildly
  different scales.
- **Model family.** A **CNN** — the input is images, and convolutions keep the spatial layout
  and find lane-line-type patterns wherever they appear (an MLP that flattens pixels fumbles
  this). The head is a single `Linear` outputting **one continuous value**, and I'd squeeze
  the `(N,1)` output to `(N,)` before the loss — the course warned broadcasting silently
  makes an `(N,N)` matrix otherwise.
- **Loss.** `nn.MSELoss` — this is **regression**, not classification. The steering angle is
  a continuous number, so there are no classes for CrossEntropy to even score; targets are
  floats, no `long()`, no one-hot. If the human-driver labels have outliers (weird jerky
  moments), use `nn.SmoothL1Loss` (Huber) instead so the squared error doesn't explode.
- **Training loop.** The Day 28b skeleton: for each epoch — `model.train()`, loop batches with
  zero_grad → forward → loss → backward → step; then `model.eval()` and evaluate on the
  validation set under `torch.no_grad()`. **Monitor validation MAE** ("on average how many
  units off is the steering" — accuracy is meaningless for a continuous output). **Keep the
  best checkpoint by val MAE** using `copy.deepcopy(model.state_dict())` — a plain reference
  would keep pointing at the live weights. Use Adam/AdamW with a learning rate (maybe warmup),
  and save model + optimizer state + epoch so a crash doesn't lose the run.
- **Export.** `model.eval()`, then export — `torch.export.export(model, (example_frame,))`
  (or `torch.jit.trace` + save). This makes a portable artifact that doesn't need my Python
  training code.
- **On-device lowering.** Lower the exported graph and compile it to a **`.pte`** file for
  the **ExecuTorch** C++ runtime — the embedded computer has no Python; the fixed
  ahead-of-time graph lets it plan memory once and keep latency predictable, and a delegate
  can offload chunks to an NPU/DSP if the box has one.
- **Quantization.** Dynamic int8 quantization on the Linear layers (the cheapest flavor)
  first; move to static or QAT only if the validation MAE degrades too much. ~4× smaller and
  faster — matters a lot on a battery/embedded real-time device.
- **Two verifications before trusting the artifact:**
  1. **Fidelity** — run the deployed (exported + quantized) artifact and the trained model on
     the same fresh inputs and check the outputs match (`torch.allclose` for the export;
     small relative error and held validation MAE for the quantized one).
  2. **Speed** — measure inference latency **on the deployed artifact on the target device**:
     warmup passes first, then time many runs with `time.perf_counter()`, report the
     **median**. Compare against the frame budget: at 20 Hz that's `1000/20 = 50 ms` per
     frame; the model meets the budget if median latency ≤ that.

## Q19 — train-function skeleton

Pseudocode (my Python is rusty, this is the shape):

```python
def train(model, train_loader, val_loader, E, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()          # or CrossEntropyLoss for classes
    best_metric = worst_possible
    best_state = None

    for epoch in range(E):
        model.train()                          # mode toggle 1
        for X, y in train_loader:
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()

        model.eval()                           # mode toggle 2
        with torch.no_grad():
            metric = evaluate(model, val_loader)   # val accuracy or val MAE

        if metric is better than best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())   # <-- the safety line

    return best_state
```

The two mode toggles are `model.train()` before the training pass and `model.eval()` before
scoring. The one line that makes keeping-the-best safe is
`copy.deepcopy(model.state_dict())` — without the deepcopy, `best_state` is just a reference
to the live weights and the next epoch silently overwrites your "best."

## Q20 — validation honesty

(a) Because a great score on the training data always looks great and **means nothing** — the
model has already seen those exact samples, so a high training score only tells you it can
reproduce what it memorized, not that it works on new data. The whole point of holding out
validation data is to score on data the model never trained on.

(b) Because the last epoch isn't necessarily the best one — the model can keep getting better
on the training data while its **validation** score gets worse in later epochs (I believe
this is called overfitting). Keeping the best-val epoch keeps the version that actually
generalizes best.

(c) The #1 bug the course warned about (Day 22, repeated on Day 23) is **forgetting to zero
the gradients** — `backward()` accumulates into `.grad`, so without `zero_grad` the gradients
pile up and training goes haywire. (Day 28b added that the #1 *evaluation* bug is scoring on
the training set and calling it "accuracy.")

## Q21 — capstone safety & workflow

(a) All openpilot development and testing happens **on your own computer against
recorded/simulated data** — replaying logged routes; no car or comma device needed. You must
**never test unreviewed code on a real vehicle**.

(b) A first PR should **avoid safety-critical vehicle-control logic** entirely. Good first
contributions: tooling and dev-experience fixes, docs/typo fixes, tests, improved error
messages, and small well-scoped bug fixes. And follow the repo's `CONTRIBUTING.md` to the
letter — its rules override generic advice.

(c) The new test must **fail before the fix and pass after** it. You write the failing case
first (reproduce, don't guess), then make the minimal change that turns it green — that's
what pins the behavior and makes the change trustworthy to reviewers.
