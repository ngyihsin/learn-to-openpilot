# Gap Probe — Key & Rubric (graders only)

Grade for correct substance; partial credit per bullet. This probe covers material the
curriculum does NOT yet teach — the per-bullet pattern of losses is the point. Also record,
per answer, whether the student reached the idea via transfer from taught material (note
which lesson they cited) or missed it entirely.

## Section A — Temporal context

**A1 (4).**
- (1.5) Why impossible: relative speed is a *rate of change of position/scale across time*;
  one frame carries no motion information (two different speeds can produce the identical
  frame). Accept any clear "you need at least two samples in time to measure a rate."
- (1.25) Way 1, e.g. **frame stacking**: concatenate the last N frames along the channel
  dimension — input becomes `(N·C, H, W)`; the CNN's first conv reads motion as cross-channel
  differences. (Accept frame differencing, optical-flow input, 3D conv.)
- (1.25) Way 2, a genuinely different mechanism, e.g. **recurrent state**: run the CNN per
  frame to an embedding, feed embeddings to an RNN/GRU/LSTM (or temporal transformer over
  the last N embeddings) that carries state across time steps.

**A2 (3).**
- (1) The runtime must now **carry the hidden state between calls** (an extra input/output
  tensor round-tripped every 50 ms) — the model is no longer a pure function of the frame.
- (1) Export must capture the stateful signature: `torch.export` needs the state as an
  explicit input/output (fixed shapes), since the graph itself can't hold mutable state.
- (1) The state must be **reset** at the start of each drive/route (and after gaps), or the
  model carries stale context into a new scene.

**A3 (3).**
- (1.5) Shuffling individual frames destroys the temporal ordering the model must learn
  from — a sample must be a *sequence*; its frames can no longer be drawn independently.
- (1.5) The unit of shuffling becomes the **sequence/clip** (windows of consecutive frames):
  shuffle clips, keep order *within* a clip.

## Section B — Data augmentation

**B1 (4).**
- (2) The steering label must be **negated** (angle → −angle): mirroring the scene mirrors
  the correct steering response; keeping the label silently teaches the model to steer the
  wrong way on half the data. (Full credit requires the negation *and* the why.)
- (1) Label-safe example: brightness/contrast/color jitter, small noise — photometric changes
  don't alter the correct steering.
- (1) Label-affecting example: horizontal translation/crop/rotation (shifts the lane's
  apparent position → steering answer changes). Accept any geometric transform with a correct
  explanation.

**B2 (3).**
- (2) Training set only.
- (1) Validation must measure performance on *undistorted, realistic* inputs with the course's
  honest-evaluation rule (never tune/score yourself on modified val data); augmenting val
  changes the thing you're measuring.

**B3 (3).**
- (1.5) In the Dataset's `__getitem__` (or a transform invoked there), so each epoch's pass
  re-rolls the randomness per sample.
- (1.5) Pre-computing once fixes a single augmented copy — the model sees the same finite set
  every epoch, so the effective-infinite-data / regularization benefit collapses.

## Section C — QAT

**C1 (4).**
- (1) Naming: static quantization and **QAT** are the stronger flavors; QAT recovers the most.
- (2) Mechanism: during training, **fake-quantize** ops simulate int8 rounding/clamping on
  weights (and activations) in the forward pass while gradients still flow (straight-through)
  — the model *trains against* the quantization error.
- (1) Why it recovers accuracy: the weights adapt to be robust to int8 precision, instead of
  being quantized after the fact with no chance to compensate.

**C2 (3).**
- (1.5) Calibration estimates the **ranges/scales (and zero-points)** used to map floats to
  int8.
- (1.5) About the **activations** — their ranges depend on real data, so representative
  inputs are run through the model to observe them (weights need no data; they're known).

**C3 (3).**
- (1.5) QAT happens **during training** (or fine-tuning) — *before* export/lower/quantize;
  the export then carries quantized ops.
- (1.5) Concrete loop change: prepare the model with fake-quant modules
  (insert observers/fake-quant, then train some epochs, then convert). Accept "wrap/prepare
  the model before the epoch loop + a convert step after" at this level.

## Score sheet

| Q | Max | S2 | S3 |
|---|-----|----|----|
| A1 | 4 | | |
| A2 | 3 | | |
| A3 | 3 | | |
| B1 | 4 | | |
| B2 | 3 | | |
| B3 | 3 | | |
| C1 | 4 | | |
| C2 | 3 | | |
| C3 | 3 | | |
| **Total** | **30** | | |

## Decision rubric (for the loop report, not for grading)

The next lesson goes to the gap with the worst **danger-weighted** showing, judged on:
1. **Failure severity** — lost points, especially *confidently wrong* answers (worse than
   "don't know").
2. **Safety/goal criticality** — would the misconception ship a wrong AV model? (e.g. B1's
   un-negated flip label silently trains wrong-way steering.)
3. **Prerequisite fit** — how much of the lesson the curriculum already supports.
4. **Buildable/gradeable** — can it be a CPU-fast pytest lesson in the course's style?
