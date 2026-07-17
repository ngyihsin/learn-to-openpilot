# Gap Probe — Grading (S2, S3)

Grader stance: strict. Every rubric bullet had to be substantively present; partial credit at
half-bullet granularity; confidently-wrong (unhedged) claims lose the bullet and are flagged as
dangerous errors. Per the key's instruction, each answer is also tagged **transfer** (reached
the idea from taught material, lesson cited) or **miss**.

## 1. Score table

| Q | Max | S2 | S3 |
|---|-----|----|----|
| A1 | 4 | 4.0 | 4.0 |
| A2 | 3 | 3.0 | 3.0 |
| A3 | 3 | 3.0 | 3.0 |
| **A subtotal** | **10** | **10.0** | **10.0** |
| B1 | 4 | 4.0 | 4.0 |
| B2 | 3 | 3.0 | 3.0 |
| B3 | 3 | 3.0 | 3.0 |
| **B subtotal** | **10** | **10.0** | **10.0** |
| C1 | 4 | 3.5 | 3.5 |
| C2 | 3 | 3.0 | 3.0 |
| C3 | 3 | 3.0 | 3.0 |
| **C subtotal** | **10** | **9.5** | **9.5** |
| **Total** | **30** | **29.5** | **29.5** |

## 2. Per-question notes and lost bullets

### Section A — Temporal context

**A1 — S2: 4/4 (transfer).** Impossibility bullet fully present ("relative speed is a *rate of
change* … needs at least two samples in time"). Way 1 = frame stacking with the exact concrete
change (`Conv2d(3*k, …)`, `__getitem__` returns frames `i-k+1 … i`, `torch.cat`) — cites
day24/day25. Way 2 = per-frame CNN encoder + Day 25e `TransformerEncoderBlock` over the last k
tokens, a genuinely different (stateful) mechanism, accepted by the key as "temporal
transformer over the last N embeddings". Transfer sources cited: day24, day25e, day25, day28c.

**A1 — S3: 4/4 (transfer).** Impossibility via v = dx/dt and the ambiguity argument ("Any
single frame is consistent with the lead car going faster than us or slower") — full credit;
transfer from EE background plus course. Way 1 = channel stacking with the concrete
`Conv2d(4, 8, ...)` change and `__getitem__` pseudocode (transfer from Day 24 hints). Way 2 =
CNN-per-frame tokens into the Day 25e transformer block, correctly locating "compare frame t
with t−1" in the attention sublayer (transfer from Day 25e).

**A2 — S2: 3/3 (transfer).** All three bullets: runtime carries state ("explicit input and
output of the exported graph … `(output, new_state) = model(frame, old_state)`"), export must
capture it with fixed shape/dtype (Day 26/27 cited), reset at new drive ("a stateful one that
is never reset will confidently blend two unrelated drives"). Bonus insight (unrequired):
fidelity check must become sequence-level; latency budget re-check.

**A2 — S3: 3/3 (transfer).** All three bullets present: buffer must live outside the fixed
graph / be fed back as an extra input each call (transfer from Day 27's "plan its memory once"),
state must be an explicit fixed-shape input or tracing misses it (transfer from Day 26's
"records one run" warning), reset at drive start with a sensible first-frames strategy. The
hedges ("I think", "probably") are appropriately placed and the substance is correct, so no
deduction; also correctly re-raises the 50 ms budget.

**A3 — S2: 3/3 (transfer).** Both bullets: shuffling frames destroys the temporal signal (a
sample must be a window), and the unit becomes the clip — "let the DataLoader shuffle the
*windows*", order kept inside. Unprompted extra: split train/val **by drive** to avoid
near-duplicate leakage, correctly transferred from Day 28b's honest-evaluation rule.

**A3 — S3: 3/3 (transfer, partly self-derived).** Both bullets: "frame 503 followed by frame
88 … the relative-speed label no longer matches the input", and "Shuffle between clips, never
inside a clip" with the `__getitem__`-returns-a-clip construction. Student flags the clip idea
as self-derived from Day 25's shuffle rationale — that is transfer working as intended.

### Section B — Augmentation

**B1 — S2: 4/4 (transfer + reasoning).** Negation with the why (2/2): "negate the label
(`steer → −steer`)" plus the mechanism of the failure mode ("with 50% flipping the left/right
evidence cancels … predict ~0 steering everywhere" — a correct and sharp account of the silent
defect). Safe example: brightness/contrast/noise (1/1). Label-affecting: horizontal translation
tied to Day 28c's offset-based label (1/1).

**B1 — S3: 4/4 (transfer + reasoning).** Sign flip with the why (2/2): "the correction must now
go the other way … every flipped sample is actively poisoned", grounded in Day 28c's label
definition and the [-1,1] range. Safe: brightness/contrast/noise, with a nice Day 23b
`channel_normalize` cross-check (1/1). Label-affecting: horizontal shift with correct
offset reasoning (1/1).

**B2 — S2: 3/3 (transfer).** Training set only (2/2); honest-eval justification from Day 28b
(1/1): val must answer "how well does this model do on data it will actually meet?". Extra
correct points: best-val checkpointing becomes noise, val must stay fixed for comparability.

**B2 — S3: 3/3 (transfer).** Training set only (2/2); honest-eval justification (1/1): "your
val MAE is measured on a distribution you invented … same crime as scoring on training data",
plus the checkpoint-comparability point. Both reasons cited to Day 28b/28c.

**B3 — S2: 3/3 (transfer).** In `Dataset.__getitem__` with fresh randomness per fetch (1.5);
pre-computing collapses the benefit to a fixed finite set the model re-memorizes (1.5).
Correct bonus: `num_workers > 0` moves augmentation compute into loader workers.

**B3 — S3: 3/3 (transfer, partly self-derived).** `__getitem__` with correct pseudocode —
notably it re-flips the label inside the augmentation (`x = flip(x); y = -y`), showing B1 was
internalized (1.5); frozen-copy argument ("2× the data, and then nothing more") (1.5).

### Section C — QAT

**C1 — S2: 3.5/4 (transfer). Lost: 0.5 on the naming bullet (blank-ish omission).**
The key's first bullet requires naming *both* stronger flavors — "static quantization and QAT".
S2's C1 contrasts QAT only against dynamic/post-training quantization and never names static
within the answer ("Post-training quantization (what `quantize_dynamic` does) …"); "QAT recovers
the most" is present ("Day 28 ranks it as the most work and the most accuracy recovered"), so
half the bullet is earned. Classified **blank-ish omission**, not confident-wrong — nothing
false is asserted, and static is correctly handled in C2. Mechanism (2/2): fake-quantize in the
forward ("snap them to the int8 grid, and convert back to float"), loss computed on the
quantized behavior, float master copy still updated by backward/step. Why-it-recovers (1/1):
"the optimizer walks the weights to a place that scores well *under int8* … QAT *trains it to
be* good."

**C1 — S3: 3.5/4 (transfer). Lost: 0.5 on the naming bullet (blank-ish omission).**
Same defect: C1 names dynamic and QAT but never names static as the other stronger flavor in
the answer body (the citation line gestures at "三種 flavors 那一段" but the answer does not
state it). "Recovers the most accuracy" is present. Blank-ish omission, not dangerous.
Mechanism (2/2): round weights (and activations) to the 256 int8 levels in the forward, loss
includes the quantization error, normal backward/step continues. The student explicitly flags
the one thing they don't know — how the gradient crosses the rounding staircase — and guesses
"the framework fakes/ignores it somehow", which is a correctly-hedged approximation of
straight-through estimation; no deduction, and this is a model example of hedging an unknown
rather than asserting it. Why-it-recovers (1/1): "the weights drift to places where rounding
hurts least … nobody is doing gradient descent anymore [after PTQ], so the error just lands on
your MAE."

**C2 — S2: 3/3 (transfer + reasoning).** Ranges/scales bullet (1.5): "record the typical value
ranges … pick each layer's activation scale/zero-point once and for all" (zero-point included).
Activations-not-weights bullet (1.5): weights readable from the checkpoint, activations depend
on input, hence representative data. Also correctly distinguishes dynamic's on-the-fly scales.

**C2 — S3: 3/3 (transfer + reasoning).** Both bullets: scale factors mapping the observed float
range onto int8's 256 levels (1.5); about activations, weights need no data (1.5), quoting
Day 28's exact sentence and adding a correct ADC full-scale analogy (too wide = coarse steps,
too narrow = clipping) from EE background.

**C3 — S2: 3/3 (transfer).** QAT during training, before export/lower (1.5), with the revised
pipeline spelled out. Concrete loop change (1.5): "before training, wrap/prepare the model …
insert fake-quantize steps", loop body untouched, convert after training — exactly the key's
accepted "prepare before the epoch loop + convert after". Honestly flags not knowing the real
API names rather than inventing them as fact.

**C3 — S3: 3/3 (transfer).** QAT "sits inside the train step, at the front", later quantize
step "now cheap/harmless" (1.5). Concrete change (1.5): `qmodel = prepare_for_qat(model)`
before the loop with the forward going through the wrapped model; explicitly labels the
function name as invented. Convert-after is present at pipeline level.

## 3. Transfer analysis per section

**Section A (temporal): transfer carried both students essentially all the way.** Neither had
been taught temporal models, yet both assembled correct architectures from taught parts:
Day 24's channel semantics → frame stacking (both students independently derived the
`Conv2d(N·C, …)` change); Day 25e's transformer block → a sequence-over-frame-features model;
Day 26/27's export/runtime constraints → all three stateful-deployment complications; Day 25's
shuffle rationale → the clip-as-unit fix. S2 additionally transferred Day 28b's
honest-evaluation rule into an untaught but correct split-by-drive leakage argument. No
transfer failure observed in this section for either student.

**Section B (augmentation): full transfer, zero losses.** The safety-critical item — negating
the flipped steering label — was derived correctly by both from Day 28c's label definition
("the steering value that would center the car"), with both students articulating the silent
wrong-way-steering defect the rubric worries about. Day 25's Dataset/DataLoader split carried
both to the `__getitem__` placement and the per-epoch-freshness argument; Day 28b's
overfitting/honest-eval material carried the train-only decision. The curriculum's existing
abstractions were sufficient scaffolding for this whole section.

**Section C (QAT): transfer carried ~95%; the only friction was recall of Day 28's taxonomy,
plus API-level knowledge the course genuinely never showed.** Both students reconstructed the
QAT mechanism from one Day 28 sentence ("simulates int8 during training so the model learns
around the error") plus Day 22/23/33 training-loop and gradient-descent material — including
the correct why-it-recovers argument. Both correctly identified calibration as
activation-range/scale estimation from Day 28's one-line description. The two half-point
losses are the same omission: neither restated "static" as the second stronger flavor inside
C1. The genuine, hard edges of the gap showed up only as *acknowledged* unknowns, not errors:
S3 could not explain gradients through rounding (straight-through estimation), and neither
student knows the real prepare/convert QAT API — both said so explicitly instead of
fabricating. Those are the pieces a QAT lesson would actually have to teach.

## 4. Dangerous errors (confident-wrong only)

**None found in either paper.** Every point lost was an omission, not a false assertion. Both
students hedged precisely where their knowledge ran out (S2: "I don't know the exact names —
I'd have to look them up"; S3: systematic 用猜的 / "I guess" tags on every extrapolated claim),
and all unhedged claims checked against the key were correct. In particular, both got the
highest-severity safety item (B1 flip-label negation) fully right, including the failure mode
of omitting it.

## 5. Grader note for the decision loop

Under the decision rubric this probe produced no danger-weighted signal separating the three
gaps by *failure*: A = B = C ≈ ceiling for both students, no dangerous errors. The
discriminating signal is instead the *acknowledged-unknown* residue, which sits entirely in
Section C (straight-through gradients, real QAT prepare/convert API) — i.e., the curriculum's
concepts transfer to QAT reasoning but not to QAT *implementation*.
