# Strict Adversarial Re-Grade — Round 1 (S2, S3)

Re-graded from scratch against `assessment/answer_key.md` under the STRICT stance: every
rubric bullet must be substantively and explicitly present; hedges earn credit only when the
exact substance is fully there; wrong/contradictory claims lose their bullet unless the
student clearly commits to the correct answer; exact API names, formulas, shapes, and file
formats were checked character-by-character. Scores below are independently re-derived, not
copied from the first grader.

---

## 1. Per-question re-derivation

### Student S2

| Q | Max | Strict | 1st grader | Justification |
|---|-----|--------|------------|---------------|
| 1 | 5 | 5 | 5 | Steps match the key's canonical order exactly; "accumulates (adds) into it" (1); pile-up/stale-direction mechanism named (1). No deduction. |
| 2 | 4 | 4 | 4 | `with torch.no_grad():` (2); "autograd would record the update step as part of the computation graph — bookkeeping, not part of the model's computation" (2). The leaf-tensor in-place aside is factually correct, so nothing to strike. |
| 3 | 4 | 4 | 4 | (a) `(2, 5, 4)` with rule; (b) error, "3 vs 4: not equal and neither is 1"; (c) `(C, 1, 1)`, keepdim + broadcast over H, W. All exact. |
| 4 | 4 | 4 | 4 | `(out_features, in_features)`; `y = x @ W.T + b`; `(N, out_features)`. Exact. |
| 5 | 4 | 4 | 4 | Raw logits; integer class labels, long dtype; wrong additions = softmax + one-hot. Exact. |
| 6 | 4 | 4 | 4 | Central-difference formula written exactly; float64 with the cancellation reason; "two full forward passes per parameter, per step" vs one backward. Exact. |
| 7 | 5 | 5 | 5 | All five shapes correct incl. `? = 128`; flatten→Linear size-mismatch bug; `x.flatten(start_dim=1)`. Exact. |
| 8 | 3 | 3 | 3 | Weight sharing/translation (1.5) + locality/spatial hierarchy (1.5) — two valid key bullets, substantively argued. |
| 9 | 4 | 4 | 4 | `__len__`/`__getitem__` returning `(features, label)`; `[4, 4, 2]`; order-bias rationale. Exact. |
| 10 | 4 | 4 | 4 | Decoupled weight decay (1.5); formula `base * min(1.0, (t + 1) / warmup_steps)` with `+1` and noisy-random-start rationale (1.5); `optimizer.param_groups[0]['lr']` (1). Exact. |
| 11 | 5 | 5 | 5 | Name→tensor dict; import-path rot; optimizer-state jolt; in-place mutation + missing/unexpected-keys report, not a new model. All four bullets explicit. |
| 12 | 4 | 4 | 4 | `requires_grad = False` → no grad → step applies nothing; fresh Linear trainable by default; openpilot shared backbone + task heads (path, lead car, lane lines). Exact. |
| 13 | 5 | 5 | 5 | Python-tied module vs portable artifact (2); trace runs once/records ops, `if x.sum() > 0` baked to one branch, silent (2); eval() bakes dropout/batchnorm inference behavior (1). Exact. |
| 14 | 4 | 4 | 4 | Need `torch.jit.load` + runtime; don't need class definition/training code (2); `torch.allclose` on same fresh inputs (2). Exact. |
| 15 | 6 | 6 | 6 | C++ runtime, no Python (1.5); export → `.pte` → on-device runtime (2); delegate = offload to accelerator (1.5); AOT optimization, fixed memory layout, no interpreter → real-time predictability (1). Exact. |
| 16 | 4 | 4 | 4 | `torch.export.export(model, (x,))` with tuple called out (2); shape specialization → matching-shape inputs only (2). Exact. |
| 17 | 6 | 6 | 6 | int8, ~4× because 8/32 bits (1.5); exact `quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` (1.5); dynamic/static/QAT, QAT best (1.5); measures size(+speed) and output error/accuracy (1.5). Exact. |
| 18 | 9 | 9 | 9 | All 8 stages present with a *why*, incl. both discriminators: MSE-because-continuous/regression, CE wrong (2); fidelity (allclose/relative error on fresh frames) AND on-device latency vs frame-rate deadline (1). No stage missing its why. |
| 19 | 6 | 6 | 6 | Epoch loop + five steps; per-epoch val with best-so-far comparison; `model.train()`/`model.eval()` + `torch.no_grad()`; `copy.deepcopy(model.state_dict())` with live-reference rationale; returns `best_state`. No logic error (`best_val = -1.0` is consistent with the stated accuracy/negated-loss score). |
| 20 | 5 | 5 | 5 | Memorization vs generalization (2); overfitting, val peaks then degrades (1.5); (c) commits to "scoring yourself on the training set and calling it 'accuracy'" as the answer for a real loop, zero_grad given explicitly as a distinct aside — the commit-plus-aside exception applies (1.5). |
| 21 | 5 | 5 | 5 | Own computer, recorded/simulated data, never test unreviewed code on a real vehicle (2); avoid safety-critical control logic; docs/tooling/tests/error messages/small fixes (1.5); test must fail before / pass after (1.5). Exact. |
| **Total** | **100** | **100** | **100** | |

### Student S3

| Q | Max | Strict | 1st grader | Justification |
|---|-----|--------|------------|---------------|
| 1 | 5 | 5 | 5 | Five steps in the key's order (3); "does not overwrite `.grad` — it accumulates" (1); pile-up → bigger stale steps → haywire (1). Exact. |
| 2 | 4 | 4 | 4 | `with torch.no_grad():` (2); update is an op on `w` that must not be recorded into the graph — "bookkeeping, not part of the model's forward math" (2). Exact. |
| 3 | 4 | 4 | 4 | `(2, 5, 4)`; error with correct 3-vs-4 reason; `(C, 1, 1)` + broadcast over H, W. Exact. |
| 4 | 4 | 4 | 4 | `(out_features, in_features)`; `y = x @ W.T + b`; `(N, out_features)`. Exact. |
| 5 | 4 | 4 | 4 | Closest call: "integer class labels (a long/integer vector), not vectors" is garbled on its face, but the required substance is explicitly present — integer labels, long dtype, and "one-hot encoding the labels (it wants plain integer labels)" as the named wrong addition. No contradiction; no deduction. |
| 6 | 4 | 4 | 4 | Central differences with the written formula (1.5); float64 with rounding rationale (1); one-parameter-at-a-time → ~2M forward passes for 1M params vs one forward+backward (1.5). Exact. |
| 7 | 5 | 5 | 5 | All shapes correct, `? = 128`; final-Linear size-mismatch bug; `x.flatten(start_dim=1)`. Exact. |
| 8 | 3 | 3 | 3 | Locality/spatial structure (1.5) + weight sharing/translation invariance (1.5). Two valid bullets. |
| 9 | 4 | 4 | 4 | Both methods with returns; `4, 4, 2`; order-bias rationale. Exact. |
| 10 | 4 | 4 | 4 | AdamW decouples weight decay, same constructor args (1.5); exact warmup formula + `+1` + random-weights/noisy-gradients why (1.5); `optimizer.param_groups[0]['lr']` (1). Exact. |
| 11 | 5 | 5 | 5 | Name→tensor ordered dict; import-path rot ("#1 way checkpoints rot"); optimizer-state jolt; in-place + missing/unexpected-keys report, not a new model. Exact. |
| 12 | 4 | 4 | 4 | `requires_grad = False` → no gradient → nothing to apply (1.5); yes, trainable by default (1); backbone + heads (path, lead car, lane lines), attach a new head (1.5). Exact. |
| 13 | 5 | 5 | 5 | Python-tied module vs portable computation (2); trace-once, `if x.sum() > 0` frozen silently to one branch (2); eval() bakes dropout/batchnorm inference behavior (1). Exact. |
| 14 | 4 | 4 | 4 | `torch.jit.load` + runtime needed; class definition/training script not needed (2); both models on same fresh inputs, `torch.allclose` (2). Exact. |
| 15 | 6 | 6 | 6 | C++ runtime, no Python (1.5); export → `.pte` → run on-device (2); delegate = offload to accelerator (1.5); (d) "(I believe) memory can be laid out in advance instead of allocated on the fly" is hedged, but the exact substance (fixed memory plan) is fully stated alongside no-interpreter + AOT optimization + real-time deadline — strict hedge rule satisfied (1). |
| 16 | 4 | 4 | 4 | Exact call with tuple explained (2); matching-shape requirement afterward (2). Exact. |
| 17 | 6 | 6 | 6 | int8, 4× because 32/8 (1.5); (b) "I'm fairly sure that's the path" hedges a character-exact `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` — substance fully present, credit stands (1.5); dynamic/static/QAT, QAT best (1.5); size shrink + output closeness to float (1.5). |
| 18 | 9 | 9 | 9 | All 8 stages with whys, incl. both discriminators: regression/MSE because continuous, "no softmax, no cross-entropy" (2); fidelity (allclose / ~10% relative error on fresh frames) AND on-device speed vs camera frame-rate deadline (1). |
| 19 | 6 | 6 | 6 | Epoch loop + five steps; per-epoch val + best-so-far; both toggles + `torch.no_grad()`; `copy.deepcopy(model.state_dict())` with live-reference rationale; returns `best_state`. "is better than" is declared pseudocode — a syntax liberty, not a logic error (key: syntax slips don't lose points; exam: pseudocode fine). |
| 20 | 5 | 5 | 5 | Memorization vs generalization (2); (b) "I believe the term is overfitting" — the term IS named and the mechanism (train improves while val worsens, memorizing) is fully described; hedge rule satisfied (1.5); (c) offers both course "#1 bugs" but explicitly commits: "Day 28b's phrasing, which matches this question" → train-set self-scoring, with zero_grad labeled as the *other* lesson's bug — commit-plus-aside exception applies (1.5). |
| 21 | 5 | 5 | 5 | Recorded/simulated data on own computer, never test unreviewed code on a real vehicle (2); avoid safety-critical control; tooling/docs/tests/error messages/small fixes (1.5); fail before / pass after (1.5). Exact. |
| **Total** | **100** | **100** | **100** | |

---

## 2. Strict totals

| Student | Strict total | First-grader total | Delta |
|---------|-------------|--------------------|-------|
| S2 | **100 / 100** | 100 / 100 | 0 |
| S3 | **100 / 100** | 100 / 100 | 0 |

Both pass (pass mark 90) under strict re-grading.

## 3. Verdict

**The first grader was NOT materially too lenient.** No question's score moved by ≥1 point;
in fact no question moved at all. Every rubric bullet — including all exactness-sensitive
items (the `quantize_dynamic` call, `torch.export.export(model, (x,))` with the tuple, the
warmup formula with the `+1`, `param_groups[0]['lr']`, the `(out, in)` weight shape, the
`.pte` format, all shape-walk values, and the `copy.deepcopy(model.state_dict())` line) —
was verified character-level correct in both papers.

The candidate deductions examined and rejected, with reasons:

1. **S3 Q5** — "a long/integer vector, not vectors" is garbled, but the required substance
   (integer indices, long dtype, one-hot explicitly named as the wrong addition) is fully and
   separately stated; nothing contradicts the key.
2. **S3 Q15(d)** — "(I believe) memory can be laid out in advance" is hedged, but the
   specific rubric substance (fixed/ahead-of-time memory plan) is completely present, so the
   strict hedge rule still awards it.
3. **S3 Q17(b)** — "I'm fairly sure that's the path" hedges an exactly correct API call; the
   substance is fully there.
4. **S2/S3 Q20(c)** — both give two "#1 bugs," but both clearly commit to the rubric's answer
   (train-set self-scoring, tied to the real-loop framing of the question) and frame
   zero_grad as an aside from a different lesson — the strict rule's commit-plus-aside
   exception applies.
5. **S3 Q19** — "if val_score is better than best_acc" is English pseudocode; the exam
   permits pseudocode and the key penalizes only logic errors, of which there are none.

The suspicious uniformity of the 100s is real but is explained by the papers themselves:
both are near-verbatim reproductions of the rubric's content (same examples, same API
strings, same numbers), so identical perfect scores are the arithmetically forced outcome,
not evidence of grader leniency.
