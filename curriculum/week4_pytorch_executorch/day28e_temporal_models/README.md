# Day 28e — Temporal Models: Seeing Motion

> **Week 4 · PyTorch / ExecuTorch** — one frame tells you where things are; only *time* tells
> you where they're going.

## Why today matters

Every model you've trained so far looks at **one frame**. But the quantities that keep a car
safe are *rates*: the lead car's closing speed, how fast the lane is drifting, whether that
pedestrian is stepping off the curb. A rate is a change **across time** — and a single frame
contains exactly zero information about it. Two scenes with wildly different speeds can
produce pixel-identical frames. No bigger network fixes this; the information simply isn't
in the input. (This is why openpilot's driving model consumes a *temporal buffer* of frames,
not a snapshot.)

Today you prove that on a synthetic task — a lane line sliding sideways at some speed, where
the label is the **speed** — and then fix it twice, with the two standard mechanisms:

1. **Frame stacking** — feed the last `T` frames as `T` input *channels*: input goes from
   `(N, 1, H, W)` to `(N, T, H, W)`, and `Conv2d(T, ...)`'s very first layer can read motion
   as differences across channels. The model stays a plain CNN — Day 24 with a wider front
   door. Simple, stateless, and the frames must travel together.

2. **Recurrent state** — embed each frame into a vector (your Day 24 CNN, minus the head),
   and feed the sequence to a **`nn.GRUCell`** that carries a hidden state `h` from step to
   step. The state is the model's *memory*: after each frame it summarizes everything seen
   so far. One frame in, one prediction out, memory carried forward — the shape of every
   streaming model.

The recurrent version comes with a deployment string attached, and you'll build it the
deployment-honest way from the start. Day 27 taught that an exported graph is a **pure
function** — it cannot hold mutable state. So your `step(frame, h) → (prediction, h_new)`
takes the state as an **explicit input** and returns it as an **explicit output**; the caller
(the runtime, at 20 Hz) round-trips `h` between calls and **resets it when a new drive
starts**. Get this signature right here and `torch.export` has nothing to complain about
later.

One more habit must change: **you can no longer shuffle individual frames.** A training
sample is now a *clip* (a window of consecutive frames — the order inside is the signal).
Day 25's shuffle rule still applies, but the unit of shuffling becomes the clip: shuffle
clips freely, never the frames within one.

## Learning goals

By the end you can:

- Explain *why* a single-frame model cannot estimate a rate — and demonstrate it with a
  trained model whose val MAE never beats guessing.
- Build a **frame-stacked** CNN (`T` frames as `T` channels) that reads motion.
- Build a **recurrent** regressor around `nn.GRUCell` with an explicit-state
  `step(frame, h) → (pred, h_new)` and a zero **`init_state`**, and explain why that
  signature is what export/on-device streaming requires.
- Say what the unit of shuffling is for sequence data, and why.

## Do this

1. **Homework (~75 min).** Implement `build_single_frame_regressor`,
   `build_stacked_regressor`, and the `TemporalRegressor` class (`init_state`, `step`,
   `forward`) in `homework.py`. The motion dataset, `mae`, and the training loop are
   provided.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 28e
   ```
   The grader trains all three models: your single-frame model must stay **blind** (high
   MAE — that's the proof, not a failure), both temporal models must see motion (low MAE),
   `step` must be a pure function of `(frame, h)`, and `forward` must equal manually
   threading `step` yourself.

## Setup note

Week 4 needs PyTorch. CPU-only is fine — all three trainings together take ~20 s:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- The dataset yields `(clip, speed)` with clips shaped `(T, H, W)` — the DataLoader batches
  them to `(B, T, H, W)`. For the *single-frame* model, the provided training loop slices
  the **last** frame (`clips[:, -1:]`, keeping a channel dim of 1) — the model gets a fair
  chance and still can't learn the label. That's the point.
- `build_stacked_regressor` is Day 28c's CNN with one change: `nn.Conv2d(T, 8, ...)` instead
  of `nn.Conv2d(1, 8, ...)`. The `(B, T, H, W)` clip *is already* the `(B, C, H, W)` layout
  a CNN expects — time simply becomes channels.
- `TemporalRegressor.step` gets a frame shaped `(B, H, W)`; `unsqueeze(1)` it into
  `(B, 1, H, W)` for the embed CNN. Then `h_new = self.cell(embedding, h)`, prediction from
  `self.head(h_new)`. Return **both** — never stash `h` on `self`. State stored on the
  module is invisible to an exported graph; state passed in and out is just another tensor.
- `init_state(batch)` is `torch.zeros(batch, d)` — the "new drive" state. Fresh drive, fresh
  zeros: stale state from the previous route is a wrong prior about the current road.
- `forward(clips)` = `init_state`, then loop `t` over the clip calling `step`, return the
  **last** prediction. The grader checks `forward` against manual `step`-threading, so
  resist any shortcut that computes the sequence differently.
- `nn.GRUCell(d, d)` wants `(B, d)` in and gives `(B, d)` back — no time dimension; *you*
  own the loop over time. (The bigger `nn.GRU` runs the whole sequence at once; the Cell
  version is the one that matches on-device streaming, which is why you build with it.)

## Check yourself

- Two clips end on the exact same final frame but have different labels. Why does this
  single fact doom any single-frame model, regardless of size or training time?
- Frame stacking vs recurrent state: which needs all `T` frames re-sent every inference,
  and which needs only the newest frame? What does each cost at 20 Hz?
- Why must `h` be an explicit input/output of `step` rather than `self.h`? (Day 27: what
  can't a fixed exported graph hold?)
- You forget to reset the state between drives. What is the model effectively assuming
  about the new road?

## Where this shows up later

openpilot's `modeld` is exactly this pattern grown up: a vision backbone feeding a temporal
module whose state is threaded through every 20 Hz call. When you replay a route on Day 29,
the "warm-up" flicker in the first frames of a segment is the state filling from zeros —
you now know precisely what that is.

**Next:** Day 29 — the openpilot on-ramp.
