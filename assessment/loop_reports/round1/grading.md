# Grading Report — Round 1 (S1, S2, S3)

Graded strictly against `assessment/answer_key.md`, bullet by bullet, half-bullet granularity
allowed. Credit awarded for correct substance regardless of phrasing; hedged-but-correct
answers received full credit per the grading rules.

## 1. Summary table

| Q | Max | S1 | S2 | S3 |
|---|-----|----|----|----|
| 1 | 5 | 5 | 5 | 5 |
| 2 | 4 | 4 | 4 | 4 |
| 3 | 4 | 4 | 4 | 4 |
| 4 | 4 | 4 | 4 | 4 |
| 5 | 4 | 4 | 4 | 4 |
| 6 | 4 | 4 | 4 | 4 |
| 7 | 5 | 5 | 5 | 5 |
| 8 | 3 | 3 | 3 | 3 |
| 9 | 4 | 4 | 4 | 4 |
| 10 | 4 | 4 | 4 | 4 |
| 11 | 5 | 5 | 5 | 5 |
| 12 | 4 | 4 | 4 | 4 |
| 13 | 5 | 5 | 5 | 5 |
| 14 | 4 | 4 | 4 | 4 |
| 15 | 6 | 6 | 6 | 6 |
| 16 | 4 | 4 | 4 | 4 |
| 17 | 6 | 6 | 6 | 6 |
| 18 | 9 | 9 | 9 | 9 |
| 19 | 6 | 6 | 6 | 6 |
| 20 | 5 | 5 | 5 | 5 |
| 21 | 5 | 5 | 5 | 5 |
| **Total** | **100** | **100** | **100** | **100** |

All three students **pass** (pass mark: 90).

## 2. Questions where any student lost ≥0.5 points

**None.** No student lost 0.5 points or more on any question. Notes on the closest calls,
all of which earned full credit under the stated grading rules:

- **Q1 (step order).** All three placed `zero_grad` fifth while noting the Day-23
  zero-grad-first rotation; the key explicitly accepts either rotation. No deduction.
- **Q5, S3** — "integer class labels (a long/integer vector), not vectors" is slightly
  garbled, but the intent (indices, not one-hot vectors) is unambiguous, and S3 separately
  names both wrong additions ("no softmax, no one-hot"). Full credit.
- **Q15(d), S3** — "(I believe) memory can be laid out in advance" is hedged but factually
  correct (fixed memory plan), alongside correct AOT-optimization and no-interpreter points.
  Hedged-but-correct → full credit per the rules.
- **Q17(b), S3** — "I'm fairly sure that's the path" hedges an exactly correct
  `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` call.
  Full credit.
- **Q17(d).** All three chose size + output-error/accuracy as the two measurements (S1 and
  S2 also mentioned speed); this matches the key's primary answer. Full credit.
- **Q20(c).** S1 and S3 mention both course "#1 bugs"; per the grading rules the rubric's
  wanted answer (scoring on the training set and calling it "accuracy") is present and was
  graded — in each case it is given as the primary answer for a real training loop. Full
  credit for all three.
- **Q18.** All eight rubric stages verified present with a *why* in each paper, including
  both discriminators: the 2-pt regression-loss bullet (MSE because steering angle is
  continuous, CrossEntropy wrong — all three) and the 1-pt dual verification bullet
  (fidelity via allclose/relative error on fresh frames, plus on-device latency vs the
  frame-rate deadline — all three).
- **Q19.** All three include the epoch loop with the five training steps, per-epoch
  validation with best-so-far comparison, both mode toggles plus `torch.no_grad()`, and the
  `copy.deepcopy(model.state_dict())` safety line with the correct live-reference rationale,
  returning `best_state`. No logic errors; pseudocode liberties (S3's `is better than`) are
  syntax, not logic.

## 3. Error patterns

**None observed.** No question saw a point loss from any student, so there are no recurring
error themes in this round. The only cross-student pattern worth recording for the
experiment log is stylistic, not substantive:

- All three papers are near-isomorphic in structure and content (same examples, same
  course-day citations, same closest-call hedges), which is consistent with all three
  having mastered — or been generated from — the same course materials. This uniformity is
  a validation signal about the curriculum's coverage, not a grading deduction.
