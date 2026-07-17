# Gap Probe Re-sit — Grading (S3, Section A only)

Grader stance: strict, same as the first round. Every rubric bullet must be substantively
present in the answer body; half-bullet granularity; confidently-wrong claims lose the bullet
and are flagged. All 來源 citations were verified against the actual curriculum files
(`curriculum/week4_pytorch_executorch/day28e_temporal_models/README.md`,
`day27_executorch/README.md`, `day25_dataloaders/README.md`).

## 1. Score table

| Q | Max | S3 first attempt | S3 re-sit |
|---|-----|-----------------|-----------|
| A1 | 4 | 4.0 | 4.0 |
| A2 | 3 | 3.0 | 3.0 |
| A3 | 3 | 3.0 | 3.0 |
| **Total** | **10** | **10.0** | **10.0** |

Same ceiling total both times. As with the Section C re-sit, the score hides the real change:
the first attempt reached 10/10 by *transfer and admitted guessing* (its 來源 lines carried
「組合方式是用猜的」, 「reset 細節是用猜的」, 「半用猜的」); the re-sit reaches 10/10 by
*citation of the new Day 28e lesson*, and every quoted fragment in its 來源 lines was found
at the cited location.

## 2. Per-question notes

### A1 — 4/4 (all three bullets, now grounded)

- **Why impossible (1.5/1.5).** "Relative speed is a rate, and a rate is a change across
  time... two scenes with wildly different speeds can produce pixel-identical frames. So no
  bigger network fixes it" — verbatim-verified against Day 28e README lines 10–13. The added
  two-clips-same-final-frame doom argument is the lesson's own Check-yourself item (README
  lines 106–107). First attempt earned this bullet too, but from EE first principles
  (v = dx/dt); it is now the lesson's argument, correctly reproduced.
- **Way 1, frame stacking (1.25/1.25).** Exact concrete change: input `(N, 1, H, W)` →
  `(N, T, H, W)`, only `nn.Conv2d(T, 8, 3, padding=1)` replaces `nn.Conv2d(1, 8, ...)`;
  "the `(B, T, H, W)` clip is already the `(B, C, H, W)` layout a CNN expects — time simply
  becomes channels" is a verbatim match of the Hints (README lines 88–90); the
  first-conv-reads-motion-as-cross-channel-differences point matches README lines 19–22, as
  does the stated trade-off (stateless, but the T frames must travel together).
- **Way 2, recurrent state (1.25/1.25).** A genuinely different mechanism, exactly the key's
  example: per-frame CNN embedding → `nn.GRUCell(d, d)` carrying hidden state. The answer
  gives the buildable detail the question demands: the deployment-honest
  `step(frame, h) -> (prediction, h_new)` signature, `unsqueeze` to `(B, 1, H, W)` for the
  embed CNN, `h_new = cell(embedding, h)`, prediction from `head(h_new)`, "never store `h`
  on `self`", `init_state(batch)` = `torch.zeros(batch, d)`, and `forward(clips)` =
  init_state + loop `step` over the T frames — all verified against Hints lines 91–102.
  Notably the student *changed* Way 2 from the first attempt's transformer-block improvisation
  to the lesson's recurrent mechanism, which also sets up A2 properly.

### A2 — 3/3 (all three bullets; the first attempt's guessed area is now cited)

This was the question the decision report singled out as "純屬猜測" in the first attempt
(Day 27 then taught only the stateless case). Day 27's README now carries the promised
stateful-export paragraph (lines 20–26), and Day 28e builds the pattern — the re-sit answers
from both.

- **Export must capture the stateful signature (1/1).** State as an explicit input *and*
  output of the exported graph, because "a fixed exported graph is a pure function — it
  cannot hold *mutable* state"; "state stored on the module (`self.h`) is invisible to an
  exported graph; state passed in and out is just another tensor" — the quoted 來源 line
  condenses Day 27 README lines 20–26 accurately (verbatim fragments verified), and the
  self.h point is Day 28e Hints lines 93–94 verbatim.
- **Runtime carries/round-trips the state (1/1).** "The caller (the runtime, at 20 Hz) has
  to round-trip `h` every call... The Day 28c model needed nothing carried between frames" —
  matches Day 27 README line 23 and Day 28e README lines 33–35.
- **Reset at new drive (1/1).** Reset to `init_state` zeros at drive start; "fresh drive,
  fresh zeros"; stale state is "a wrong prior about the current road" — Hints lines 95–96
  verbatim. First attempt guessed a reset was needed and hedged on the how ("My guess is you
  zero the buffer... or fill the buffer with copies of the first frame"); the re-sit states
  the lesson's actual scheme and adds the warm-up-flicker connection (README lines 117–120,
  correctly attributed).

### A3 — 3/3 (both bullets, one now quoted from the lesson)

- **What breaks (1.5/1.5).** A sample is now a clip and "the order inside the clip is the
  signal"; shuffling individual frames scrambles it and destroys exactly the
  change-across-time information the temporal model exists to read — back to the blind
  single-frame model. Substantively the key's first bullet in full.
- **New unit of shuffling (1.5/1.5).** "The unit of shuffling becomes the clip: shuffle
  clips freely, never the frames within one", with Day 25's shuffle-every-epoch rationale
  still applying — the 來源 quote is a verbatim match of Day 28e README lines 38–41, and
  Day 25's "shuffle each epoch to avoid order bias" (README lines 9–10) checks out. First
  attempt derived clip-as-unit itself and tagged it 「半用猜的」; it is now the lesson's
  stated rule, cited.

## 3. What changed vs the first attempt

Score delta: 0 (10/10 → 10/10; Section A was already at ceiling via transfer). Substance
delta, with verified 來源 lines:

1. **A1 Way 2** — from an improvised CNN-tokens-into-Day-25e-transformer design (accepted by
   the key, but self-assembled: 「組合方式是用猜的」) to the lesson's canonical recurrent
   `GRUCell` + explicit-state `step`/`init_state` construction (Day 28e README + Hints,
   quotes verified).
2. **A2 stateful export** — the first attempt's biggest guess (hedged "I think", "I worry",
   "My guess is") is now answered from Day 27's new stateful paragraph and Day 28e's
   never-stash-`h`-on-`self` hint, with accurate verbatim citations.
3. **A2 reset semantics** — from a guessed zero-or-copy-first-frame scheme to the lesson's
   "fresh drive, fresh zeros / stale state is a wrong prior", plus the warm-up-flicker
   detail.
4. **A3 clip-as-unit** — from self-derived (「半用猜的」) to the lesson's quoted rule.

## 4. Remaining losses

**None in Section A.** No bullet lost in either attempt. Minor observations, no deduction:
the A1 embed description ("Conv → ReLU → MaxPool → Flatten → Linear(…, d) → ReLU") is a
reasonable rendering of "your Day 24 CNN, minus the head" rather than a quote, and is
consistent with the lesson; all hedges from the first attempt are gone because the material
is now taught.

## 5. Dangerous errors

**None.** Every checkable claim was verified against the curriculum files and matched. Each
來源 quotation was found verbatim (or as an accurate ellipsis-condensation) at the cited
location: Day 28e README lines 10–13, 19–28, 33–41, 88–102, 106–107, 117–120; Day 27 README
lines 20–26; Day 25 README lines 9–10. The 來源 lines are genuine citations, not
decorations, and — unlike the first attempt — contain no 用猜的 tags at all.

## 6. Grader note

This closes the Section A loop the same way Day 28d closed Section C: the first attempt
showed transfer could reach the right answers but flagged three guessed regions (Way-2
architecture assembly, stateful export mechanics, reset details); the Day 28e lesson plus
the Day 27 patch now cover all three, and the re-sit answers every one from the text with
accurate citations. Pass criterion (per the decision report's pattern: full marks and no
用猜的 sourcing) is met.
