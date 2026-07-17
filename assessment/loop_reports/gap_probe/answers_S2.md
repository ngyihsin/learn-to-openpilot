# Gap Probe — Answers (Student S2)

## A1 — Why a single frame can't give relative speed, and two ways to add temporal context

A single frame is a snapshot. From one image the model can estimate *where* the lead car is
(its position/size in the frame), but relative speed is a *rate of change* — how the lead car's
position moves between frames. Any quantity that is a derivative over time needs at least two
samples in time; one frame simply doesn't contain that information. (Same reason you can't get
velocity from a single position measurement in a control system.)

Two concrete ways to give the model temporal context:

**Way 1 — Frame stacking on the channel axis.** Instead of feeding one image of shape
`(3, H, W)`, feed the last k frames concatenated on the channel dimension: shape `(3·k, H, W)`,
e.g. k=4 gives 12 input channels. Concretely: change the first layer from `Conv2d(3, …)` to
`Conv2d(3*k, …)`, and change the `Dataset.__getitem__(i)` so that sample i returns frames
`i-k+1 … i` stacked with `torch.cat` (Day 24's shape discipline: the conv filters now span all
k frames, so they can learn "the car got bigger/closer between channel groups" — that *is*
relative speed). The model stays stateless: every call still gets everything it needs as input.

**Way 2 — A per-frame CNN encoder plus a small transformer over the last k frame-features.**
Run each frame through the Day 28c CNN backbone to get a feature vector (a "token") per frame.
Keep a rolling window of the last k tokens, shape `(1, k, d)`, and run the Day 25e
`TransformerEncoderBlock` over that sequence — self-attention lets the current frame's token
attend to the previous ones, which is exactly "compare now to a moment ago." A regression head
on the last token outputs steering + relative speed. To build it: `features = cnn(frame)` per
frame, buffer k of them, `x = block(tokens)`, `out = head(x[:, -1])`. This one carries state
between calls (the token buffer), which is the deployment cost A2 asks about.

來源: day28c_steering_regression, day24_cnns, day25e_transformer_block, day25_dataloaders

## A2 — Deployment complications of a stateful model

Versus the stateless Day 28c model:

- **The runtime must carry the state between calls.** At 20 Hz, every inference now needs the
  buffer/hidden state produced by the previous call. On-device there is no Python object to
  hold it (Day 27: ExecuTorch is a C++ runtime executing a fixed graph), so the state has to
  become an **explicit input and output of the exported graph** — call it as
  `(output, new_state) = model(frame, old_state)` and have the caller loop the state back in.
- **Export must capture it.** `torch.jit.trace` / `torch.export` record a graph for example
  inputs with fixed shapes (Day 26/27). A hidden Python-side buffer would simply not exist in
  the exported artifact, and the ahead-of-time memory plan must include the state tensor with a
  fixed shape. So the state's shape and dtype must be pinned at export time and passed in the
  example inputs.
- **A new drive needs a reset.** When a drive starts (or replay jumps routes), the state from
  the previous drive is garbage — the runtime must reinitialize it (e.g. zeros, or "not enough
  frames yet" handling for the first k−1 frames). The stateless model never had this failure
  mode; a stateful one that is never reset will confidently blend two unrelated drives.
- Also: fidelity verification (Day 26's `allclose` check) now has to compare *sequences* of
  calls, not single calls, and the latency budget (50 ms at 20 Hz) must still hold with the
  state plumbing included.

來源: day26_exporting, day27_executorch, day28c_steering_regression

## A3 — What shuffling breaks for temporal data

Day 25's advice to shuffle every epoch assumed each sample is independent. For a temporal
model a training sample is a *sequence* (a window of consecutive frames plus a label). If you
shuffle **individual frames** and then build windows from the shuffled order, the "consecutive"
frames in a window come from different moments or even different drives — the temporal signal
(how things move between adjacent frames) is destroyed, and the model is being asked to read
relative speed out of noise. The stateful variant is even worse: its carried state assumes the
next input really is the next moment in time.

The **unit of shuffling should become the sequence/clip** (or the whole drive): pre-cut the
logged drives into windows of k consecutive frames, keep the frames *inside* each window in
order, and let the DataLoader shuffle the *windows*. You still get the anti-order-bias benefit
Day 25 wanted (batches aren't all from one stretch of one drive), without breaking time inside
a sample. I'd also split train/validation **by drive**, in the spirit of Day 28b's honest-
evaluation rule: neighboring frames are near-duplicates, so putting frame i in train and frame
i+1 in val would leak and inflate the val score.

來源: day25_dataloaders, day28b_training_project

## B1 — Horizontal flip and the steering label

Flipping the frame horizontally mirrors the scene: a lane line that was offset to the left is
now offset the same amount to the right. The steering command that centers the car must
therefore mirror too — **negate the label** (`steer → −steer`). If you flip the image but keep
the label, you've manufactured training pairs that teach exactly the wrong response, and with
50% flipping the left/right evidence cancels — the cheapest way to hit that contradiction is to
predict ~0 steering everywhere, so the model learns to do nothing.

- **Safe without touching the label:** photometric changes — random brightness/contrast (or a
  little pixel noise). Making the image darker doesn't move the lane, so the correct steering
  is unchanged; it just makes the model robust to lighting.
- **Would also require changing the label (besides flipping):** a horizontal **translation** of
  the image — shifting the frame sideways by p pixels moves the lane offset, which is exactly
  the quantity the steering label encodes (Day 28c's dataset labels the lane's horizontal
  offset), so the label must be adjusted by the corresponding amount.

來源: day28c_steering_regression, 自行推理 (label geometry)

## B2 — Augment train, validation, or both?

**Training set only.** The course's evaluation rule (Day 28b) is that the validation number
must honestly answer "how well does this model do on data it will actually meet?" — the #1
evaluation bug is reporting a number that looks good but means nothing. The car will see real,
un-flipped, un-jittered frames, so val must be real frames. Two extra reasons from the same
lesson: (1) checkpointing keeps the *best-val* epoch, and a randomly-augmented val set makes
val MAE jump around for reasons that have nothing to do with the model, so "best" becomes
noise; (2) val scores across epochs/runs must be comparable, so the val set should be fixed.
Augmentation is a *training* trick to fight overfitting; evaluation stays clean.

來源: day28b_training_project, day28c_steering_regression

## B3 — Where augmentation lives in the pipeline

It belongs **inside the `Dataset.__getitem__`** (applied at the moment a sample is fetched),
with fresh randomness on every call. Day 25 split the pipeline into `Dataset` (how to get
sample i) and `DataLoader` (batching/shuffling/prefetch): since the DataLoader calls
`__getitem__` again every epoch, a random flip/brightness draw inside it means epoch 1 and
epoch 2 see *different* versions of the same underlying frame — the model effectively trains
on an unbounded stream of variants it can't memorize. Bonus: with `num_workers > 0` the
augmentation compute happens in the loader's parallel workers, so it doesn't starve the model.

Augmenting the dataset **once, ahead of time** turns that unbounded stream back into a fixed,
finite set — say each frame plus one flipped copy. After a few epochs the model has seen every
one of those files many times and can start memorizing them again, which is precisely the
overfitting we were fighting (Day 28b: training keeps improving while val quietly degrades).
You'd also multiply your storage for far less diversity. The benefit of augmentation is mostly
in the *per-epoch freshness*, and only on-the-fly augmentation gives you that.

來源: day25_dataloaders, day28b_training_project

## C1 — How QAT works and why it recovers accuracy

Post-training quantization (what `quantize_dynamic` does) takes a model whose weights were
optimized assuming full float32 precision and then rounds them onto a coarse int8 grid. The
model never got a vote — whatever error the rounding introduces just lands on top of the
finished weights, and for a sensitive regressor that can double the MAE, as here.

**QAT** moves the quantization *into* training. Mechanically: during the forward pass you
"fake-quantize" — take the float weights (and activations), snap them to the int8 grid, and
convert back to float — so the loss is computed on the outputs the model would produce *as an
int8 model*. The backward pass and optimizer step still update the underlying float weights
(you keep a float copy as the thing being trained; the quantized version is derived from it
each forward pass). So the Day 22/23 loop is unchanged — forward, loss, backward, step,
zero_grad — except the forward now includes the rounding error.

Why that recovers accuracy: gradient descent minimizes whatever loss you show it (Day 33). If
the loss already contains the quantization error, the optimizer walks the weights to a place
that scores well *under int8*, e.g. settling near values that survive rounding, instead of a
float-optimal point that happens to round badly. Post-training quantization hopes the rounded
model is still good; QAT *trains it to be* good. That's why Day 28 ranks it as the most work
and the most accuracy recovered.

來源: day28_quantization, day22_tensors_autograd, day33_gradient_descent

## C2 — What calibration is for in static quantization

Mapping float values to int8 needs a **scale** (and offset): you have 256 levels, so you must
decide what float range they cover. For **weights** that's easy at any time — the weights are
fixed numbers sitting in the checkpoint, so you can read their min/max directly. But
**activations** (the values flowing between layers) depend on the *input*: you can't know how
big a layer's outputs get by looking at the model alone. Dynamic quantization dodges this by
computing activation scales on the fly at each inference — flexible but with runtime overhead.

**Static** quantization instead fixes the activation scales ahead of time, and that's what the
calibration set is for: run a small batch of representative inputs through the model, record
the typical value ranges the activations actually take at each layer, and use those observed
ranges to pick each layer's activation scale/zero-point once and for all. So: calibration
estimates the **ranges (hence scale factors) of the activations** — the weights don't need it.
It's the same reason Day 28c said to measure latency on realistic conditions: you can only get
these numbers by running real data through, not by inspecting the model.

來源: day28_quantization, 自行推理 (why activations need sample data)

## C3 — Where QAT sits in the pipeline, and the concrete loop change

The Week 4 pipeline was **train (22–24) → export (26) → lower (27) → quantize (28)**, with
quantization as a post-training last step. QAT moves the quantization decision to the **front**:
it happens *during the training stage*, so the pipeline becomes roughly
**train-with-fake-quant → export → lower → convert to the real int8 model**. (Day 27 noted
quantization is often done as part of the lowering — with QAT, the lowering/conversion step is
where the fake-quantized model becomes an actual int8 artifact; the learning about int8 already
happened upstream.)

The concrete change to the Day 28c loop: **before training, wrap/prepare the model so its
forward pass simulates int8** — insert fake-quantize steps on the weights and activations of
the layers you'll quantize. The loop itself is untouched: same
`opt.zero_grad() → loss = MSELoss(model(X), y) → loss.backward() → opt.step()`, same best-val-
MAE checkpointing — only now `model(X)` goes through simulated quantization, so the MAE you
validate and checkpoint on is the *int8* MAE. After training you convert the prepared model to
a genuinely quantized one and continue to Day 26/27's export/lowering, then re-run the two
Day 28c verifications (fidelity vs. the trained model, latency vs. the 50 ms frame budget).
On API: the course only showed me `torch.ao.quantization.quantize_dynamic`; I'd expect QAT to
have a prepare-then-convert pair in the same `torch.ao.quantization` namespace, but I don't
know the exact names — I'd have to look them up.

來源: day28_quantization, day28c_steering_regression, day27_executorch
