# Gap Probe Re-sit — Grading (S3, Section C only)

Grader stance: strict, same as the first round. Every rubric bullet must be substantively
present in the answer body; half-bullet granularity; confidently-wrong claims lose the bullet
and are flagged. All 來源 citations were verified against the actual curriculum files
(`curriculum/week4_pytorch_executorch/day28d_qat_from_scratch/`, `day28_quantization/`).

## 1. Score table

| Q | Max | S3 first attempt | S3 re-sit |
|---|-----|-----------------|-----------|
| C1 | 4 | 3.5 | 3.5 |
| C2 | 3 | 3.0 | 3.0 |
| C3 | 3 | 3.0 | 3.0 |
| **Total** | **10** | **9.5** | **9.5** |

Same total. The score hides the real change: the substance moved from
correct-but-reconstructed guessing to grounded, verifiable citation of Day 28d — but the one
half-point lost the first time was lost again, for the identical omission.

## 2. Per-question notes

### C1 — 3.5/4. Lost: 0.5 on the naming bullet (same omission as first attempt).

- **Naming (0.5/1).** The key requires naming *both* stronger flavors — static and QAT — and
  that QAT recovers the most. The re-sit's C1, like the first attempt's, never names
  **static** anywhere in the answer body; it contrasts QAT only against "PTQ" generically.
  "QAT trained at the same 3 bits recovers most of that damage" earns the recovers-half.
  Static is again handled correctly in C2, so this is again a blank-ish omission, not
  dangerous — but under the strict stance the bullet is still only half present. This is the
  one bullet the Day 28d lesson did not repair, because Day 28d is a QAT deep-dive and the
  taxonomy lives in Day 28; the student answered from the new lesson and didn't restate it.
- **Mechanism (2/2), now grounded.** First attempt reconstructed the mechanism from one
  Day 28 sentence and explicitly could not explain how gradients cross `round()` ("I guess
  the framework fakes/ignores it somehow. 用猜的"). The re-sit states the full mechanism with
  the exact Day 28d recipe (`qmax = 2**(bits-1) - 1`, `scale = |x|.max() / qmax`,
  round/clamp/dequantize — verified against README Hints line 74), names and explains the
  **straight-through estimator** (verified against README lines 19–25), the two-static-method
  `autograd.Function` construction (Hints lines 78–80), and float "shadow weights" quantized
  on the fly (Hints lines 82–84). The first attempt's acknowledged unknown is now a cited,
  correct explanation. 來源 line checks out in full.
- **Why it recovers (1/1).** "The rounding error is a surprise the model never got to react
  to... the weights 'learn to be quantized': they settle where the grid can represent them
  well" — direct, verified quote of README lines 16–17, with the PTQ contrast intact. The
  3-bit / ~7× MAE figure matches README lines 11–12 exactly.

### C2 — 3/3 (unchanged score, firmer grounding).

- **Ranges/scales (1.5/1.5).** "The ranges (and from those, I believe the scales) of the
  activations" — the hedge on "scales" is honest and immediately grounded: the 來源 line
  attributes "scales" to Day 28's "static (also pre-computes activation scales from a small
  calibration set)", which is a verbatim match of `day28_quantization/README.md` line 14.
  (Zero-points not mentioned; the key treats them as parenthetical, and Day 28d's scheme is
  symmetric — no deduction, consistent with the first-attempt grading.)
- **Activations, not weights (1.5/1.5).** Direct verified quote of Day 28d Hints lines
  89–92 ("runs representative inputs through the model to estimate **activation** ranges
  (weights need no data — you already have them)"), plus the correct pre-computed-vs-on-the-fly
  contrast with dynamic. The ADC analogy is now *in the lesson itself* (same Hints bullet) —
  what was an EE-background extrapolation the first time is now curriculum-confirmed.

### C3 — 3/3 (unchanged score; the invented API is replaced by the real one).

- **Where QAT sits (1.5/1.5).** "During training — before export and lowering, at the very
  start of the train → export → lower → quantize pipeline", citing the learning goal "it
  happens **during training**, before export/lower" — verified, README lines 43–44.
- **Concrete loop change (1.5/1.5).** First attempt invented `prepare_for_qat(model)` and
  flagged the name as made up. The re-sit gives the actual Day 28d construction: replace
  `nn.Linear` with `QATLinear` whose forward uses `STEQuant.apply(self.weight, self.bits)`
  via `nn.functional.linear(x, w_q, self.bias)` while the stored weight stays float —
  verbatim match of Hints lines 81–84 — and correctly notes the loop itself is untouched,
  quoting the `train_regressor` docstring "Works unchanged for float models AND QAT models —
  that's the point" (verified, `homework.py` line 62). The jumpy-QAT-loss / judge-by-val-MAE
  note matches Hints lines 87–88. This is the key's "prepare the model with fake-quant
  modules before the loop" bullet, delivered at a *more* concrete level than the key demands.

## 3. What changed vs the first attempt

Went from guessed/unknown to grounded (with verified 來源 lines):

1. **C1 STE / gradient-through-rounding** — the first attempt's only genuine acknowledged
   unknown ("用猜的 for that part") is now a full, correct, cited explanation (Day 28d README
   "Why today matters", STE paragraph).
2. **C1 mechanical recipe** — from "round to the 256 levels" hand-waving to the exact
   scale/round/clamp/dequantize formula and shadow-weight scheme (Day 28d Hints).
3. **C3 API/loop change** — from an admittedly invented `prepare_for_qat` name to the real
   `QATLinear` + `STEQuant.apply` construction and the verified "loop works unchanged"
   docstring (Day 28d Hints + homework.py line 62).
4. **C2 ADC analogy** — previously personal EE transfer, now confirmed as the lesson's own
   framing (Day 28d Hints).

Remaining loss:

- **C1 naming bullet (−0.5), a repeat.** Static is still never named in the C1 answer body.
  Not dangerous (nothing false asserted; static is correct in C2), but it is the same
  half-bullet lost the same way twice, and it comes from Day 28's taxonomy rather than the
  new Day 28d material.

## 4. Dangerous errors

**None.** Every checkable claim in the re-sit was verified against the curriculum files and
matched, including the numeric claim (3-bit PTQ ≈ 7× worse val MAE) and every quoted phrase
in the 來源 lines. The one hedge ("I believe the scales", C2) is both flagged and correct.
The 來源 lines are accurate citations, not decorations: each quoted fragment was found
verbatim (or near-verbatim) at the cited location.

## 5. Grader note

The re-sit demonstrates exactly what the Day 28d lesson was built to close: the first
attempt's Section C residue was "concepts transfer, implementation unknown" (STE, real QAT
mechanics/API). All of that residue is now answered from the lesson with accurate citations.
The only remaining defect is exam craft, not knowledge: not restating Day 28's flavor
taxonomy inside C1.
