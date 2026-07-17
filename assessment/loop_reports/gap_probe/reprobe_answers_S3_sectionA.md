# Gap Probe — Section A answers (Student S3, re-sit)

## A1 — Why one frame can't give relative speed, and two ways to fix it (4 pts)

Why a single-frame model fundamentally cannot do this: relative speed is a **rate**, and a
rate is a change **across time**. A single frame contains exactly zero information about it —
the lesson said two scenes with wildly different speeds can produce pixel-identical frames.
So no bigger network fixes it; the information simply is not in the input. If two clips end
on the same final frame but have different speed labels, a single-frame model must give the
same answer to both, so it can never beat guessing.

Two concrete ways to give the model temporal context (from Day 28e):

**Way 1 — Frame stacking.** Feed the last `T` frames as `T` input *channels*. The input goes
from `(N, 1, H, W)` to `(N, T, H, W)`, and you change only the first layer of the Day 28c
CNN: `nn.Conv2d(T, 8, 3, padding=1)` instead of `nn.Conv2d(1, 8, ...)`. The `(B, T, H, W)`
clip is already the `(B, C, H, W)` layout a CNN expects — time simply becomes channels, so
the very first conv layer can subtract frames across channels and read motion. It stays a
plain CNN: simple and stateless, but the T frames must travel together into every inference.

**Way 2 — Recurrent state.** Embed each frame into a vector with the CNN minus its head
(Conv → ReLU → MaxPool → Flatten → Linear(…, d) → ReLU), then feed the sequence to a
**`nn.GRUCell(d, d)`** that carries a hidden state `h` from step to step. The state is the
model's memory: after each frame it summarizes everything seen so far. You write it the
deployment-honest way: `step(frame, h) -> (prediction, h_new)` — unsqueeze the frame to
`(B, 1, H, W)` for the embed CNN, `h_new = cell(embedding, h)`, prediction from
`head(h_new)`, return **both**, never store `h` on `self`. `init_state(batch)` is
`torch.zeros(batch, d)`. One frame in, one prediction out, memory carried forward — only the
newest frame is needed each call. For training, `forward(clips)` does `init_state` then
loops `step` over the T frames in order and returns the last prediction.

來源: Day 28e README — "A rate is a change across time — and a single frame contains exactly
zero information about it. Two scenes with wildly different speeds can produce
pixel-identical frames"; the two numbered mechanisms (frame stacking, recurrent state) and
the Hints about `Conv2d(T, 8, ...)`, `step`, `init_state`, and `GRUCell`.

## A2 — Deployment complications of keeping state at 20 Hz (3 pts)

The recurrent version (Way 2) keeps state between calls, and Day 27 says a fixed exported
graph is a **pure function** — it cannot hold *mutable* state. So versus the stateless Day
28c model:

1. **Export must capture the state explicitly.** The hidden state has to be an **explicit
   input and output** of the exported graph — the `step(frame, h) -> (pred, h_new)`
   signature. State stored on the module (`self.h`) is invisible to an exported graph;
   state passed in and out is just another tensor, so `torch.export` has nothing to
   complain about.
2. **The runtime must now carry the state between calls.** The caller (the runtime, at
   20 Hz) has to **round-trip `h` every call**: take the `h_new` that came out of this tick
   and feed it back in as `h` on the next tick. The Day 28c model needed nothing carried
   between frames.
3. **The state must be reset when a new drive starts.** The caller resets `h` to the
   `init_state` zeros on a fresh drive — "fresh drive, fresh zeros." Stale state from the
   previous route is a wrong prior about the current road. (Day 28e also mentions the
   warm-up flicker at the start of a segment is just this state filling from zeros.)

來源: Day 27 README — "a fixed graph cannot hold mutable state … must expose that state as
an explicit input and output of the exported graph — the runtime round-trips it every call,
and the caller resets it when a new drive starts"; Day 28e README/Hints — never stash `h`
on `self`, `init_state` = zeros, stale state is a wrong prior.

## A3 — What goes wrong shuffling individual frames; the new unit of shuffling (3 pts)

For sequence data a training sample is no longer a frame — it is a **clip**, a window of
consecutive frames, and **the order inside the clip is the signal**. If you keep shuffling
individual frames, you scramble that order, so the change-across-time information (the very
thing the temporal model exists to read, like the speed) is destroyed — the model would be
back to having no rate information, like the blind single-frame model.

Day 25's shuffle-every-epoch rule still applies (shuffling avoids order bias), but the
**unit of shuffling becomes the clip**: shuffle clips freely, never the frames within one.

來源: Day 28e README — "A training sample is now a *clip* … the order inside is the signal.
Day 25's shuffle rule still applies, but the unit of shuffling becomes the clip: shuffle
clips freely, never the frames within one"; Day 25 README on why we shuffle each epoch.
