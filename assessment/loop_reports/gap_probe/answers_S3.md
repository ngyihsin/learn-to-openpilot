# Gap Probe — Answers (S3)

Honest note first: the course did not really teach temporal models or augmentation details or
how QAT works inside, so a lot of this is me reasoning from the sentences I remember plus my
EE background. I answer everything anyway like the instructions say.

---

## A1 — Why one frame can't give relative speed, and two ways to add time

From physics (this part I trust, it is EE not ML): speed is change of position over time,
v = dx/dt. One camera frame is one sample at one instant. From a single photo you can maybe
tell *where* the lead car is (its position / how big it looks), but you cannot tell how that
position is *changing*. Any single frame is consistent with the lead car going faster than us
or slower than us. So the information is simply not in the input — no training trick fixes
that, the model would just learn some average.

Two ways to give the model temporal context:

**Way 1 — stack the last N frames as channels.** Day 24 built `Conv2d(1, 8, 3, padding=1)`
where the 1 is the input channel count. So instead of feeding one grayscale frame shaped
`(1, H, W)`, keep the last, say, 4 frames and feed `(4, H, W)` — change the conv to
`Conv2d(4, 8, ...)`. The rest of the Day 28c pipeline (CNN → one continuous output, MSELoss,
val MAE) stays the same. The Dataset's `__getitem__(i)` would return frames `i-3 .. i` stacked
together plus the label at time `i` (pseudocode: `x = stack(frames[i-3:i+1]); y = rel_speed[i]`).
Now the network can "see" the lead car in two positions and the difference between the frames
carries the speed.

**Way 2 — treat a sequence of frames as tokens into a transformer block.** Day 25e built a
`TransformerEncoderBlock` where tokens attend to each other with `softmax(QK^T/√d)V`, input
shaped `(B, T, d)`. My idea: run the CNN part on each of the last T frames to get one feature
vector per frame, so you get a `(B, T, d)` sequence, then put that through the encoder block
(attention → residual → norm → feed-forward), then a Linear head that outputs the one number.
The attention sublayer is the only place tokens look at each other (the lesson said this), so
that is exactly where "compare frame t with frame t−1" would happen. A classmate builds it as:
`features = [cnn(f) for f in last_T_frames]; seq = stack(features); out = block(seq); pred = head(out[last token])`.

來源: Day 24 hints (Conv2d(1,8,3,padding=1) 的 channel 數), Day 25e 主文 (「(B, T, d)」、「the feed-forward runs on each token independently... tokens only look at each other in attention」), Day 28c 主文; 物理部分是 EE 常識; 組合方式是用猜的.

## A2 — Deployment complications of keeping state at 20 Hz

My proposals both need the runtime to remember the last N frames between calls (a frame
buffer), so I treat that buffer as the "state". Compared to the stateless Day 28c model:

- **What the runtime must carry:** Day 27 said the ExecuTorch runtime plans its memory *once*,
  ahead of time, and runs a fixed graph. Now, on top of that, something has to hold the last
  N frames (or the memory tensor) *between* calls, 20 times per second. I think the `.pte`
  file only captures the computation graph, so the buffer probably has to live outside the
  graph, in the C++ runtime code around the model — the caller feeds the state back in as an
  extra input each call. That is extra plumbing that Day 28c did not need.
- **What export must capture:** `torch.export.export(model, (example_input,))` records one run
  and specializes on the example's shapes. So the state tensor has to be an explicit input
  (with a fixed shape) so the trace sees it. If the state is hidden inside the module as a
  Python variable, I worry the "run once and record" style export misses it, the same way
  Day 26 warned tracing can silently miss data-dependent logic.
- **When a new drive starts:** the state must be reset, otherwise the first frames of drive 2
  get compared against the last frames of drive 1 and the model computes a nonsense "speed".
  My guess is you zero the buffer at drive start (similar spirit to how the training loop must
  `zero_grad` so old stuff doesn't accumulate — old state piling up would also make things go
  haywire), or fill the buffer with copies of the first frame so the model sees "no motion"
  instead of garbage.

Also the 50 ms frame budget from Day 28c still applies but now the model does N frames worth
of work per call, so latency must be re-measured.

來源: Day 27 主文 (「plan its memory once」「fixed, ahead-of-time graph」), Day 27 hints (「export 會 specialize on the example's shapes」), Day 26 (tracing records one run 的限制), Day 22 hints (gradients pile up 的比喻), Day 28c (50 ms frame budget); reset 細節是用猜的.

## A3 — What breaks if you shuffle individual frames

Day 25 said we shuffle every epoch so the model doesn't learn the *order* of the data (order
bias). But for the temporal model, the order *inside a short window* IS the signal — the model
needs frame t and frame t−1 next to each other to see motion. If you shuffle individual
frames, a "sample" would be built from frames that are not actually consecutive (frame 503
followed by frame 88), so the apparent motion is random garbage and the relative-speed label
no longer matches the input. You would be destroying exactly the information we added in A1.

Fix: the unit of shuffling should become the **whole sequence/clip**, not the frame. Build the
Dataset so `__getitem__(i)` returns one complete short clip (the N consecutive frames plus its
label) as one sample, keeping the frames inside it in time order — then let the DataLoader
shuffle *which clip* comes when. Shuffle between clips, never inside a clip.

來源: Day 25 主文與 Check... 不對, 我沒讀 Check yourself — Day 25 主文 (「shuffle each epoch to avoid order bias」, Dataset 需要 `__len__` 和 `__getitem__`); clip 當單位是我自己推的, 半用猜的.

## B1 — Horizontal flip and the steering label

If you mirror the frame left-right, the lane line that was offset to the left is now offset to
the right by the same amount. Day 28c said the label is "the steering value that would center
the car", so the correction must now go the other way: the label's **sign must flip**
(`y → −y`). If you flip the image but keep the label, you are teaching the model to steer
*toward* the wrong side — every flipped sample is actively poisoned. (Assuming steering 0 is
straight and the range is symmetric like the [-1, 1] mentioned in Day 28c.)

- **Safe without touching the label:** change the brightness/contrast a bit, or add a little
  pixel noise. The lane line stays in the same horizontal place, so the correct steering is
  unchanged. (Day 23b's `channel_normalize` would partly undo a brightness shift anyway, which
  makes me more confident it is label-neutral.)
- **One more that WOULD require changing the label (besides flipping):** shift/translate the
  image horizontally by k pixels. Now the lane line is at a different offset, so the correct
  "center the car" steering value is different — you would have to adjust the label to match
  the new offset (by however many units k pixels corresponds to).

來源: Day 28c 主文 (「a lane line at some horizontal offset, labeled with the steering value that would center the car」, MAE 0.03 with steering in [-1,1] 的例子), Day 23b (channel_normalize); flip/shift 的推理是用猜的 (課程沒教 augmentation 細節).

## B2 — Augment train, val, or both?

Training set only. Two reasons from the course:

1. The stated purpose is "to fight overfitting" — overfitting is a *training* disease
   (Day 28b: past some point the model keeps getting better on the training set while
   validation quietly degrades). Augmentation gives training more variety; validation's job
   is not to be learned from, so it gains nothing from variety.
2. Honest evaluation. Day 28b hammered that the #1 evaluation bug is scoring on the training
   set — the val number is supposed to answer "how well does this do on real, held-out data?"
   Real camera frames from the car will not be randomly flipped. If you augment val, your
   val MAE is measured on a distribution you invented, not on the real task, so the number
   stops meaning anything — same crime as scoring on training data, just a different way of
   making the number dishonest. Also if val changes randomly each run, "keep the best
   checkpoint by val MAE" (Day 28c) would be comparing epochs on different tests.

來源: Day 28b hints (「The #1 evaluation bug is scoring on the training set... Always report val」, overfitting 的段落), Day 28c (checkpoint the best epoch by val MAE).

## B3 — Where in the pipeline does augmentation go?

Put it inside the Dataset's `__getitem__` (the "how to get sample i" method from Day 25) —
i.e., the random flip/brightness happens fresh at the moment a sample is fetched:

```
__getitem__(i):
    x, y = load frame i, label i
    if random() < 0.5:  x = flip(x); y = -y
    return x, y
```

The DataLoader asks the Dataset for every sample once per epoch, so sample i gets a *new*
random decision every epoch — epoch 1 maybe flipped, epoch 2 not, epoch 3 flipped plus
brighter. Over many epochs the model effectively sees an endless stream of variations.

If instead you augment the dataset once ahead of time (make one fixed augmented copy on disk),
each sample is frozen to one particular variation forever. The model sees the exact same
pictures every epoch — that is precisely the situation where it can start memorizing them
(overfitting again). You'd get, say, 2× the data, and then nothing more, instead of new
variety every epoch. Doing it per-`__getitem__` costs almost nothing extra and keeps the
variety infinite-ish.

來源: Day 25 主文 (Dataset = 「how to get sample i」, `__len__` + `__getitem__`; DataLoader 負責 batch/shuffle), Day 28b (overfitting); 「每個 epoch 重新隨機」的機制是我推的, 半用猜的.

## C1 — How QAT works and why it recovers accuracy

What the course literally told me (Day 28): QAT "simulates int8 *during* training so the model
learns around the error — the most work, recovers the most accuracy." Here is my best
mechanical picture built on that sentence:

During each training forward pass, you take the weights (and I think the activations too) and
push them through a pretend-int8 step — round them to the nearest of the 256 levels int8 can
represent, then continue the forward pass with those rounded values. So the loss the network
sees already *includes* the quantization error. Then the normal loop from Day 22/23 runs:
loss → backward → optimizer step. Because gradient descent is always nudging weights downhill
on whatever loss you show it, and this loss is "the error *of the quantized version*", the
weights drift to places where rounding hurts least — e.g. two weights that individually round
badly can shift so the rounded network still computes nearly the right answer. The model
"learns around" the rounding, exactly like the sentence says.

Why post-training quantization can't recover this: dynamic quantization (Day 28) rounds an
already-finished model. The float weights were optimized assuming full precision; rounding
them afterwards is an error the training never got a chance to compensate for — nobody is
doing gradient descent anymore, so the error just lands on your MAE. With QAT the compensation
happens while the knobs are still being turned.

(One thing I genuinely don't know: rounding is a staircase function, and from Day 33's picture
the gradient is a slope — a staircase is flat almost everywhere, so I am not sure how the
gradient gets through the rounding step. I guess the framework fakes/ignores it somehow.
用猜的 for that part.)

來源: Day 28 主文 (三種 flavors 那一段, 「simulates int8 during training so the model learns around the error」, int8 = 4× smaller), Day 33 (gradient descent 只會照你給它的 loss 往下走); 機械細節是用猜的.

## C2 — What calibration is for

Day 28 said static quantization "also pre-computes activation scales from a small calibration
set". So calibration is about the **activations**, not the weights. The weights are already
sitting there after training — you can look at them directly and see their range, no extra
data needed (that's why dynamic can quantize weights ahead of time with nothing). But the
activations — the in-between values flowing through the network — depend on what input you
feed in, so you cannot know their range just by staring at the model.

Calibration = run a small representative set of inputs through the float model and record how
big the activations at each layer actually get (their typical range / min-max). The numbers
being estimated are the **scale factors**: how to map that float range onto int8's 256 levels.
From my EE side this is exactly choosing the full-scale range of an ADC: set the range too
wide and you waste your 8 bits (coarse steps), too narrow and real values clip. Dynamic
quantization skips this by measuring the activation range on the fly at inference time, each
call; static measures it once, ahead of time, on the calibration set, and bakes it in.

來源: Day 28 主文 (「static (also pre-computes activation scales from a small calibration set)」, dynamic = 「weights quantized ahead of time, activations on the fly」); ADC 類比是 EE 常識; 其餘用猜的.

## C3 — Where QAT sits in the pipeline, and the change to the Day 28c loop

The Week 4 pipeline was: train → export (Day 26) → lower (Day 27) → quantize (Day 28), with
quantize as the last step. QAT breaks that ordering: because the "aware" part happens *during
training*, QAT sits **inside the train step**, at the front — you train with quantization
simulated, and then the later quantize step is just making real the int8 the model already
trained for. So the pipeline becomes roughly: train-with-fake-int8 → export → lower →
quantize (now cheap/harmless because the model already expects it).

Concrete change to the Day 28c training loop: the loop skeleton stays Day 28b's
(forward → loss → backward → step → zero_grad, val MAE each epoch, deepcopy best state_dict).
The one line that changes is the **forward pass**: instead of `pred = model(x)`, the forward
runs through the quantize-simulating version of the model, something like

```
qmodel = prepare_for_qat(model)        # wraps layers so forward fake-rounds to int8
for epoch: for batch:
    pred = qmodel(x)                   # forward WITH simulated int8 error
    loss = MSELoss(pred.squeeze(-1), y)
    loss.backward(); opt.step(); opt.zero_grad()
```

I don't know the real PyTorch function name for the prepare step (Day 28 only showed
`quantize_dynamic`), so that name is invented — but the point is: same loop, same MSE/MAE,
only the forward now includes the pretend-int8 rounding, so the optimizer trains around it.

來源: Day 28 主文與結尾 (「train (22–24) → feed data (25) → export (26) → lower (27) → quantize (28)」, QAT 「simulates int8 during training」, 「often done as part of this lowering」 in Day 27→28), Day 28b/28c (訓練迴圈骨架); API 名稱是用猜的.
