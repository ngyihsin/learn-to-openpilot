# Final Exam — Answers (Student S3)

## Q1 — the five-step training loop

The five steps, in order:

1. forward (run the model on the batch: `out = model(X)`)
2. loss (score how wrong it is: `loss = criterion(out, y)`)
3. backward (`loss.backward()` — computes the gradients)
4. update (`opt.step()` — the optimizer nudges the weights downhill)
5. zero the gradients (`opt.zero_grad()`)

(The Day 23 lesson actually wrote `zero_grad` first in the loop, but it's the same cycle — as long as the grads are zeroed once per step it works out.)

(a) You have to zero every step because `loss.backward()` does not overwrite `.grad` — it **accumulates** into it. Each call adds the new gradient on top of whatever is already sitting there.

(b) If you forget, the gradients pile up across steps, so your update is no longer "the gradient of this batch's loss" — it's the sum of every gradient since the start. The steps get bigger and bigger in stale directions and training goes haywire. The lessons called this the #1 beginner bug.

## Q2 — which context manager for the manual update

`with torch.no_grad():`

You need it because autograd records every operation on tensors that require grad. The weight update `w -= lr * w.grad` is itself an operation on `w`, and you do NOT want that update recorded as part of the computation graph — it's bookkeeping, not part of the model's forward math. Wrapping it in `no_grad()` tells autograd not to track it.

## Q3 — broadcasting

The rule I learned: align the shapes from the right, pad the shorter one with 1s on the left, then compare dimension by dimension — each pair must be equal or one of them must be 1, and the output takes the bigger one.

(a) `(2, 1, 4)` with `(5, 4)`: pad the second to `(1, 5, 4)`. Compare: 2 vs 1 → 2, 1 vs 5 → 5, 4 vs 4 → 4. Result: **(2, 5, 4)**.

(b) `(3,)` with `(4,)`: compare 3 vs 4 — not equal, and neither is 1. **Error.**

(c) With `dim=(1, 2), keepdim=True` the mean has shape **(C, 1, 1)** — keepdim keeps the reduced axes as size-1 instead of dropping them. Then `img - mean` works because (C, 1, 1) broadcasts cleanly against (C, H, W): the C's match and the two 1s stretch out over H and W. Without keepdim the mean would be shape (C,) and the right-alignment would line C up against W, which is wrong.

## Q4 — nn.Linear weight shape

`nn.Linear` stores its weight as **(out_features, in_features)** — that's the PyTorch convention, which is why you have to transpose it.

The one-liner: `y = x @ W.T + b`

Shapes: `(N, in) @ (in, out) → (N, out)`, and the bias `(out,)` broadcasts across all N rows. Output shape: **(N, out_features)**.

## Q5 — CrossEntropyLoss requirements

The two requirements:
1. The model output must be **raw logits** — no softmax applied.
2. The target must be **integer class labels** (a long/integer vector), not vectors.

The two things beginners wrongly add:
1. Applying **softmax** to the outputs first (the loss does that internally).
2. **One-hot encoding** the labels (it wants plain integer labels).

The Day 28b hints literally said: "no softmax, no one-hot."

## Q6 — numerical gradient check

Use **central differences** — wiggle the parameter a tiny amount in *both* directions and look at how the loss changes, i.e. something like `(loss(w + eps) - loss(w - eps)) / (2 * eps)` — rather than a one-sided difference, because two-sided is more accurate. Do it in **float64** for accuracy (float32 rounding error swamps the tiny eps).

Why it's too slow for real training: you can only wiggle **one parameter at a time**. Each wiggle needs a full forward pass (two, with central differences), so a model with a million parameters needs on the order of two million forward passes just to get *one* gradient, for *one* step. Autograd gets all the gradients from a single forward + backward. So the numerical way is only useful as a correctness check — the finite-difference check the graders used — not for training.

## Q7 — CNN shape walk

Input: `(N, 1, 8, 8)`

- After `Conv2d(1, 8, kernel_size=3, padding=1)`: padding=1 keeps the spatial size, and there are now 8 channels → **(N, 8, 8, 8)**
- After ReLU: same shape → **(N, 8, 8, 8)** (ReLU doesn't change shape)
- After `MaxPool2d(2)`: halves the spatial size → **(N, 8, 4, 4)**
- After flatten (keeping the batch dim): 8 × 4 × 4 = 128 → **(N, 128)**
- So `? = 128`, and after `Linear(128, 2)` → **(N, 2)**

The #1 CNN bug this prevents: a **size mismatch at the final Linear layer** — computing the wrong input size for it because you didn't trace the shapes through the conv/pool stack.

The call that flattens everything except the batch dimension: `x.flatten(start_dim=1)`.

## Q8 — why a CNN beats an MLP on images

1. An MLP has to flatten the pixels, which throws away the spatial layout. A CNN slides small filters across the image, so it keeps and uses the 2-D structure — it detects local patterns like edges and corners.
2. The convolution shares the **same** filter weights across every position of the image. That means far fewer parameters, and the same pattern is detected no matter where in the image it appears (translation invariance) — an MLP would have to relearn "edge" separately for every location.

## Q9 — Dataset / DataLoader

(a) Exactly two methods: `__len__` (how many samples there are) and `__getitem__(i)` (returns sample i as a `(features, label)` tuple).

(b) With 10 samples and `batch_size=4`: batches of **4, 4, 2** — the last one is the ragged leftover.

(c) You shuffle each epoch to avoid **order bias** — if the data comes sorted (say all of one class first), the model sees long runs of the same thing and the gradient steps get biased by that ordering. Shuffling makes each batch a fair mix.

## Q10 — optimizers and LR schedules

(a) Adam vs AdamW: the difference is **how weight decay is applied** — AdamW *decouples* the weight decay from the gradient step. They take the same constructor arguments; the difference is which class you instantiate.

(b) Linear warmup: `lr = base * min(1.0, (t + 1) / warmup_steps)`. The `+1` is the off-by-one fix so step 0 isn't a dead zero. After `warmup_steps` it just holds at `base`. Warmup solves the problem at the very start of training: the weights are random and the gradients are large and noisy, so full-size steps right away can blow the run up. Ramping the LR up gently avoids that.

(c) `optimizer.param_groups[0]['lr']` — and if a scheduler is involved, read it *before* calling `scheduler.step()`, because the scheduler mutates that value in place.

## Q11 — checkpointing

(a) A `state_dict` is an ordered dictionary mapping every parameter and buffer *name* (like `"fc1.weight"`, `"fc1.bias"`) to its tensor. The optimizer has one too (momentum buffers, step counts, learning rate).

(b) Because pickling the whole model object hard-codes your class's import path into the file — the moment you refactor or rename the class, the checkpoint won't load anymore. The lesson called this the #1 way checkpoints rot. The state_dict is just named tensors, so it survives refactors.

(c) Because the optimizer carries its own state — momentum buffers, running averages, step counts, the current learning rate. If you resume with only the weights, the optimizer starts cold and the very next update step is wrong/jolted compared to where the run left off. Saving both (plus the epoch) lets you continue as if nothing happened.

(d) `model.load_state_dict(...)` mutates the model **in place** and returns a **report of missing/unexpected keys** — it does NOT return a new model. That trips people up.

## Q12 — transfer learning

(a) Freezing a parameter just means setting `requires_grad = False` on it. Autograd then computes **no gradient** for it, so when the optimizer steps there is nothing to apply — the parameter stays put. (That's also why an optimizer built over all parameters still leaves frozen ones alone: no grad, nothing to update.)

(b) Yes — a freshly built `nn.Linear` has `requires_grad=True` on its parameters by default, so the new head trains automatically without you doing anything.

(c) openpilot's driving model is itself a **shared backbone with multiple task heads** — path, lead car, lane lines. When the team wants a new prediction, they attach a new head instead of retraining the whole network. That's exactly the freeze-backbone / swap-head pattern.

## Q13 — why export, tracing's blind spot, eval()

(a) A trained `nn.Module` is Python objects plus Python code — to run it you'd need your whole training script and a Python interpreter, which a phone or the comma device can't spin up per camera frame. Exporting captures just the *computation* in a portable format that runs without any of that.

(b) `torch.jit.trace` runs the model **once** on your example input and records the operations that actually executed. So any **data-dependent control flow** — like `if x.sum() > 0: ... else: ...` — gets silently frozen to whichever branch the example happened to take. The trace has no idea the other branch exists. `torch.jit.script` / `torch.export` handle this better because they look at the code, not just one run.

(c) `model.eval()` bakes in inference-time behavior — things like dropout and batchnorm behave differently in training vs. inference, and you want the exported artifact to have the inference behavior permanently.

## Q14 — loading a traced model

To load and run `m.pt` elsewhere you need: `torch.jit.load("m.pt")` (and a PyTorch/TorchScript runtime to call it with). What you **don't** need is the original model class definition or your training script — the lesson emphasized "no original class definition required."

The single check that proves fidelity: run **both** the original and the exported model on the same fresh inputs and confirm the outputs match with `torch.allclose`. That's the check every deployment pipeline runs.

## Q15 — ExecuTorch

(a) ExecuTorch is PyTorch's tiny, portable **C++** runtime for executing models on-device — at inference time there is **no Python at all**.

(b) The three-stage pipeline:
1. **Export** — `torch.export` your trained model into a clean, backend-agnostic graph (an `ExportedProgram`).
2. **Lower / compile** that graph into a **`.pte`** file.
3. **Run** the `.pte` on-device with the ExecuTorch runtime, optionally offloading pieces to hardware accelerators.

(c) A *delegate* is when you offload part of the graph to specialized hardware (an accelerator) instead of running it on the general CPU runtime.

(d) With a fixed graph compiled ahead of time, the device knows exactly what it's going to compute before it runs: no Python interpreter overhead, optimizations can be done ahead of time, and (I believe) memory can be laid out in advance instead of allocated on the fly. For a real-time device that has to process every camera frame on a deadline, that predictability and low overhead is what you need.

## Q16 — torch.export.export

```python
exported = torch.export.export(model, (x,))
```

The second argument must be a **tuple** of the model's positional args — even with just one tensor, you write `(x,)`, not `x`. (Put the model in `eval()` first, and you can get a runnable module back with `exported.module()`.)

"Specializes on the example's shapes" means the exported graph is recorded for the shapes of the example input you gave it. So afterwards you have to run it with inputs of **matching shapes** — you can't just feed it a different-sized batch or image and expect it to work.

## Q17 — quantization

(a) It stores the weights as 8-bit integers instead of 32-bit floats. That makes the model roughly **4× smaller**, and the factor is 4 simply because 32 bits / 8 bits = 4. You trade a little precision for size and speed; the accuracy hit is small and usually acceptable.

(b) `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` — I'm fairly sure that's the path; it returns a quantized *copy*, and you should quantize in `eval()` mode.

(c) The three flavors: **dynamic**, **static**, and **quantization-aware training (QAT)**. Dynamic is the easiest; **QAT recovers the most accuracy**.

(d) The two things to measure: (1) the model's **size/footprint** actually shrank (the lesson measured by serializing the state_dict to a BytesIO and reading its length — should be roughly 4× smaller), and (2) the quantized model's **outputs stay close to the float version** (the grader checked within ~10% relative error). Basically: did you gain the size, and did you keep the accuracy.

## Q18 — design: steering-angle model, end to end

- **Dataset & split.** Wrap the 100k frames + steering angles in a `Dataset` (`__len__`, `__getitem__(i)` returning `(frame_tensor, angle)`). Standardize/normalize the input images (per-channel mean/std normalization, like Day 23b — helps gradient descent when inputs are on sane scales). Split into **train** and **validation** with `random_split`, with a seeded generator so it's reproducible. The validation set exists because you must never score yourself on data you trained on.
- **Model family.** A **CNN** — the input is camera images, and convolutions keep the spatial layout and detect local patterns (edges → shapes → objects) with shared filter weights. This is literally what openpilot's vision model is. The final layer would be a `Linear` down to **1 output** — the steering angle.
- **Loss function.** This is a **regression** problem, not classification — the steering angle is a single continuous number, not a category. So no softmax, no cross-entropy. Use **MSE loss** (mean of squared errors between predicted angle and the human driver's angle), the same loss from Day 33/35. MSE penalizes big misses much harder than small ones, which seems right for steering.
- **Training loop.** The standard five steps (zero_grad → forward → loss → backward → step) over a `DataLoader` with shuffling each epoch. Optimizer: Adam or AdamW (the lessons said Adam often "just works"), maybe with linear warmup at the start. **Monitor:** validation loss (val MSE) every epoch — not accuracy, since there are no classes here. **Keep:** a `copy.deepcopy` of the `state_dict` from whichever epoch has the best validation loss — the last epoch can be worse than an earlier one because the model starts memorizing the training data. Also checkpoint model + optimizer + epoch regularly with `torch.save` so a crash doesn't lose the run.
- **Export & on-device lowering.** Put the model in `eval()`, then `torch.export.export(model, (example_frame,))` to get the backend-agnostic graph, then lower/compile it to a **`.pte`** file for the **ExecuTorch** C++ runtime — the embedded computer can't run a Python + PyTorch stack per camera frame. Possibly use a delegate to offload to whatever accelerator the in-car computer has.
- **Quantization.** Quantize to int8 (dynamic quantization at minimum; QAT if the accuracy hit matters) — roughly 4× smaller and faster, which matters a lot on constrained real-time hardware.
- **Two verifications before trusting the deployed artifact:**
  1. **Fidelity** — run the exported/quantized model and the original float model on the same fresh frames and check the outputs match (`torch.allclose` for the export; small relative error, ~10% was the course's number, for the quantized version). The artifact must compute the same thing as what you validated.
  2. **Speed** — measure that it actually runs fast enough on the device to keep up with the camera frame rate (it's a real-time deadline; a model that's correct but too slow misses frames).

## Q19 — train-function skeleton

Pseudocode (rusty, sorry):

```python
import copy

def train(model, train_loader, val_loader, optimizer, criterion, E):
    best_acc = -inf          # or best val loss = +inf if regression
    best_state = None
    for epoch in range(E):
        model.train()                      # mode toggle 1
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()                       # mode toggle 2
        with torch.no_grad():
            val_score = evaluate(model, val_loader)   # val accuracy or val loss

        if val_score is better than best_acc:
            best_acc = val_score
            best_state = copy.deepcopy(model.state_dict())   # <-- the safety line

    return best_state
```

The two mode toggles are `model.train()` before the training pass and `model.eval()` before scoring (a no-op for a plain MLP but it matters once dropout/batchnorm show up). The one line that makes keeping the best state actually safe is `copy.deepcopy(model.state_dict())` — a plain reference would keep pointing at the *live* weights, so the next epoch would silently overwrite your "best."

## Q20 — validation honesty

(a) Because a great score on the training data tells you almost nothing — the model may have simply memorized those exact samples. "That number always looks great and means nothing," as the course put it. Only performance on data the model never trained on tells you it actually generalizes.

(b) Because the last epoch can be *worse* than an earlier one: past some point the model keeps improving on the training data while getting worse on validation — it's memorizing instead of learning (I believe the term is overfitting). So you keep whichever epoch scored best on validation.

(c) The course actually flags two "#1 bugs" in different lessons, so I'll give both: for a *real* training loop (Day 28b's phrasing, which matches this question) it's **scoring yourself on the training set and calling that "accuracy"** — always report validation. The other #1 bug, from Day 22, is forgetting to zero the gradients so they accumulate across steps.

## Q21 — safety & workflow

(a) All openpilot development and testing happens **on your own computer against recorded/simulated data** — replaying logged routes. You never need a car. And you must **never test unreviewed code on a real vehicle**.

(b) A first PR should **avoid safety-critical vehicle-control logic** entirely. Good first contributions are: tooling/dev-experience fixes, docs and typo fixes, tests, improved error messages, and small well-scoped bug fixes. Also: small and focused — one change, no bundled unrelated tweaks — and follow the project's `CONTRIBUTING.md` to the letter.

(c) The new test must **fail before your fix and pass after it**. Write the failing test first (reproduce before you fix) — that's what proves the test actually pins the bug and your change actually fixes it.
