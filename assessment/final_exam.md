# Final Exam — PyTorch to an Autonomous-Driving Inference Model

**Total: 100 points · Pass: 90 · Time guide: ~2.5 hours**

Rules:
- Closed book **except** the course materials in `curriculum/` (lesson READMEs, your homework,
  the solutions). No other references.
- Answer in prose and/or code as each question asks. Short and correct beats long and vague.
- Code answers may be PyTorch-flavored pseudocode unless the question says *runnable*.

---

## Section 1 — Tensors, autograd & the training loop (25 pts)

**Q1 (5 pts).** Write the canonical PyTorch training loop as five named steps, in order.
Then explain: (a) why gradients must be zeroed **every** step, and (b) what specifically goes
wrong in training if you forget — name the mechanism, not just "it breaks."

**Q2 (4 pts).** In a hand-written gradient-descent loop (no optimizer), the weight update
`w -= lr * w.grad` must happen inside a particular context manager. Which one, and why is it
required there?

**Q3 (4 pts).** Broadcasting. For each pair, give the result shape or say it errors and why:
- (a) `(2, 1, 4)` with `(5, 4)`
- (b) `(3,)` with `(4,)`
- (c) You normalize an image tensor of shape `(C, H, W)` per channel:
  `img.mean(dim=(1, 2), keepdim=True)` — what shape does the mean have, and why does the
  subtraction `img - mean` work?

**Q4 (4 pts).** `nn.Linear(in_features, out_features)` stores its weight `W` with what shape?
Write the one-line expression that applies this layer to a batch `x` of shape `(N, in)` using
only `@`, `.T`, and `+`, and give the output shape.

**Q5 (4 pts).** You train a classifier with `nn.CrossEntropyLoss`. State the two requirements
on what you pass it (the model output and the target), and name the two preprocessing steps
beginners wrongly add.

**Q6 (4 pts).** Describe how to check a gradient numerically: which difference formula, which
dtype, and why. Then explain why this method is far too slow to train a real model even
though it is correct.

---

## Section 2 — Building & feeding models (25 pts)

**Q7 (5 pts).** Shape walk. Input is a batch `(N, 1, 8, 8)`. It passes through
`nn.Conv2d(1, 8, kernel_size=3, padding=1)` → `ReLU` → `nn.MaxPool2d(2)` → flatten (keep the
batch dim) → `nn.Linear(?, 2)`. Give the shape after each stage and the correct `?`.
What is the #1 CNN bug this arithmetic prevents, and which call flattens everything except
the batch dimension?

**Q8 (3 pts).** Give two structural reasons a CNN beats a plain MLP on images.

**Q9 (4 pts).** (a) A custom `Dataset` must implement exactly which two methods, returning
what? (b) With 10 samples and `batch_size=4`, what batch sizes does a `DataLoader` yield?
(c) Why shuffle the training set each epoch?

**Q10 (4 pts).** (a) Adam vs AdamW — what is the one difference? (b) Write the linear-warmup
formula for the LR at step `t` with base LR `base` and `warmup_steps` total (mind the
off-by-one at step 0), and say what problem warmup solves. (c) How do you read the *current*
LR off a live optimizer?

**Q11 (5 pts).** Checkpointing. (a) What is a `state_dict`? (b) Why save the `state_dict`
rather than the whole model object? (c) Why must a *resume* checkpoint also include the
optimizer's state? (d) What does `model.load_state_dict(...)` return, and does it produce a
new model?

**Q12 (4 pts).** Transfer learning. (a) Mechanically, what does "freezing" a parameter mean,
and why does the optimizer then leave it alone? (b) After you replace a model's final head
with a fresh `nn.Linear`, are the new head's parameters trainable by default? (c) openpilot's
driving model relates to this pattern how?

---

## Section 3 — From training to the car: export, ExecuTorch, quantization (25 pts)

**Q13 (5 pts).** (a) Why must a model be *exported* at all before it can run on a device —
what is wrong with shipping the `nn.Module`? (b) `torch.jit.trace` has a specific blind spot:
what kind of code does it silently get wrong, and why (what does tracing actually do)?
(c) Why call `model.eval()` before exporting?

**Q14 (4 pts).** After `torch.jit.trace(model, example).save("m.pt")`, what do you need in
order to load and run `m.pt` elsewhere — and what *don't* you need? What single check proves
the exported model is faithful to the original?

**Q15 (6 pts).** ExecuTorch. (a) What is it, in one sentence (mention the language of the
runtime and what's absent at inference time)? (b) Describe the three-stage pipeline from a
trained PyTorch model to inference on-device, naming the file format. (c) What is a
*delegate*? (d) Why does a fixed, ahead-of-time graph help a real-time device?

**Q16 (4 pts).** `torch.export.export` — show the exact call for a model taking one tensor
`x` (mind the second argument's type), and explain what "the exported graph specializes on
the example's shapes" means for how you may run it afterward.

**Q17 (6 pts).** Quantization. (a) What does int8 quantization do to weights and roughly how
much smaller does the model get — and why that factor? (b) Write the one-line dynamic
quantization call for the `Linear` layers of a model. (c) Name the three flavors of
quantization and say which one recovers the most accuracy. (d) After quantizing, name the two
things you must measure to know whether it was worth it.

---

## Section 4 — Applied: an autonomous-driving inference project (25 pts)

**Q18 (9 pts).** Design task. You are given 100,000 logged, labeled camera frames from a
dashcam: each frame is paired with the **steering angle** (a single continuous value) the
human driver applied. Design, end-to-end, how you would train and deploy a model that
predicts steering angle from a frame and runs on an embedded in-car computer. Cover, in
order: dataset & split, model family, **loss function (and why — this output is not a
class)**, the training loop (what you monitor and what you keep), export & on-device
lowering, quantization, and the **two verifications** you run before trusting the deployed
artifact (fidelity and speed). Bullet points are fine; each stage needs the *what* and the
*why*.

**Q19 (6 pts).** Write the skeleton (pseudocode fine) of a `train` function that: trains for
`E` epochs over a `DataLoader`, evaluates on a validation loader every epoch, keeps the best
model state seen so far, and returns it. Include the two mode toggles and the one line that
makes "keeping the best state" actually safe.

**Q20 (5 pts).** (a) Why must the validation set never overlap the training set — what does a
great score on training data actually tell you? (b) Why keep the *best-validation* epoch
instead of the last epoch? (c) What does the course name as the #1 **evaluation** bug in real
training loops?

**Q21 (5 pts).** Safety & workflow, per the capstone. (a) Where does all your openpilot
development and testing happen, and what must you never do with unreviewed code? (b) What
kinds of changes should a first PR avoid, and what kinds make good first contributions?
(c) When fixing a bug, what property must your new test have relative to the fix?
