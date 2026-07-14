# Round 2 Grading — Students S2 and S3

Graded strictly against `assessment/answer_key.md` per-question bullet breakdowns.
Half-bullet granularity used where the grading stance requires it.

## 1. Score table

| Q | Max | S2 | S3 |
|---|-----|-----|-----|
| 1 | 5 | 5 | 5 |
| 2 | 4 | 4 | 4 |
| 3 | 4 | 4 | 4 |
| 4 | 4 | 4 | 4 |
| 5 | 4 | 4 | 4 |
| 6 | 4 | 4 | 4 |
| 7 | 5 | 5 | 5 |
| 8 | 3 | 3 | 3 |
| 9 | 4 | 4 | 4 |
| 10 | 4 | 4 | 4 |
| 11 | 5 | 5 | 5 |
| 12 | 4 | 4 | 4 |
| 13 | 5 | 5 | 5 |
| 14 | 4 | 4 | 4 |
| 15 | 6 | 6 | 6 |
| 16 | 4 | 4 | 4 |
| 17 | 6 | 6 | 6 |
| 18 | 9 | 9 | 9 |
| 19 | 6 | 6 | 6 |
| 20 | 5 | 4.25 | 4.25 |
| 21 | 5 | 5 | 5 |
| **Total** | **100** | **99.25** | **99.25** |

Both students pass (threshold 90).

## 2. Deductions (every question where any student lost ≥ 0.5 points)

### Q20(c) — S2: −0.75, S3: −0.75 (of the 1.5-pt bullet)

The exam asks: "What is the #1 bug in **real training loops** the course warns about?"
The rubric's answer (1.5 pts): "scoring on the **training set** and calling it 'accuracy.'"

Both students gave two alternatives and **committed to the wrong one**, mentioning the
rubric's answer only as a parenthetical aside. Per the grading stance ("if they commit to
the WRONG one and mention the right one only as an aside, award half the bullet"), each
earns 0.75 of 1.5.

- **S2** committed to: "The #1 bug the course warns about: **forgetting to zero the
  gradients** each step … (The course also names the #1 *evaluation* bug: scoring on the
  training set and calling it 'accuracy.')" — The zero-grad bug is the rubric's Q1 "#1
  **beginner** bug"; the Q20(c) rubric answer, training-set scoring, appears only as the
  aside.
- **S3** committed to: "The #1 bug the course warned about (Day 22, repeated on Day 23) is
  **forgetting to zero the gradients** … (Day 28b added that the #1 *evaluation* bug is
  scoring on the training set and calling it 'accuracy.')" — Same structure: wrong primary
  commitment, correct answer demoted to a parenthetical.

No other question lost ≥ 0.5 points for either student. Notes on close calls that were
awarded full credit:

- **Q17(d), both students**: listed fidelity/accuracy plus size-and-speed; the rubric wants
  size and output error and explicitly accepts speed alongside error — both measurables are
  substantively present. Full 1.5.
- **Q19, S3**: pseudocode placeholders (`worst_possible`, "is better than") are acceptable —
  the exam allows pseudocode and the comparison logic is unambiguous. Full credit.
- **Q20(b), S3**: "I believe this is called overfitting" is hedged, but the specific correct
  substance (val degrades in later epochs while train improves; keep best-val) is fully
  present, so the hedge does not cost points per the grading stance.
- **Q1, both**: presented update-then-zero ordering while noting the zero-grad-first Day 23
  rotation — the rubric explicitly accepts the rotation.

## 3. Error patterns

1. **Only recurring error (both students, identical): conflating the course's two "#1 bug"
   warnings on Q20(c).** The course names two distinct #1 bugs — forgetting `zero_grad`
   (the #1 *beginner* bug, Q1 territory) and scoring on the training set (the #1 *real
   training loop / evaluation* bug, the Q20(c) answer). Both students led with zero-grad
   and parenthesized the training-set-scoring answer, inverting the emphasis the question
   asked for. The identical structure of both answers (down to the parenthetical framing)
   suggests the course material's dual "#1 bug" labeling invites this confusion.
2. **Hedging-with-alternatives as a strategy**: both students tended to name a second
   candidate answer in parentheses (Q20(c) for both; S2's Q3(c) keepdim aside, S3's
   scheduler aside on Q10(c)). Harmless everywhere except Q20(c), where the primary
   commitment was the wrong alternative.
3. Otherwise no recurring weaknesses: shape arithmetic, broadcasting, export/ExecuTorch
   pipeline, quantization flavors, and the applied design question were answered accurately
   and completely by both students.
