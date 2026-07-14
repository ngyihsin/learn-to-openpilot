# Assessment — the Loop-Engineering Harness

This directory holds the **final exam** for the PyTorch → autonomous-driving-inference track
(Week 5 foundations → Week 4 PyTorch/ExecuTorch → the openpilot capstone), plus the
loop-engineering process that validates the curriculum itself.

## The loop

The curriculum is treated like code: it ships only when it passes its tests. The "tests" are
simulated students.

```
┌────────────────────────────────────────────────────────────┐
│  1. TEACH    students study ONLY the curriculum materials  │
│  2. TEST     they sit assessment/final_exam.md (100 pts)   │
│  3. GRADE    graded against answer_key.md + rubric         │
│  4. REVIEW   every lost point is classified:               │
│                A) curriculum gap   → fix the lesson        │
│                B) ambiguous exam   → fix the question      │
│                C) student slip     → note, no change       │
│  5. LOOP     repeat until EVERY simulated student ≥ 90     │
└────────────────────────────────────────────────────────────┘
```

Three simulated students (LLM agents constrained to the curriculum content) at different
levels sit each round:

| Student | Profile |
|---------|---------|
| **S1 — strong** | CS grad, solid Python, studies every lesson + hints carefully |
| **S2 — target** | The course's actual audience: EE/CS grad, basic Python, a little systems knowledge, **no ML background**; follows the recommended study order (Week 5 days 31–34 → Week 4) |
| **S3 — struggling** | Rusty Python, takes materials literally, relies only on what lessons state explicitly; skips "Check yourself" sections |

**Pass criterion: all three students score ≥ 90/100.** The struggling student is the point:
if S3 passes, the lessons — not prior knowledge — did the teaching.

## Files

| File | What |
|------|------|
| [`final_exam.md`](final_exam.md) | The exam students sit (closed-book except the curriculum) |
| [`answer_key.md`](answer_key.md) | Reference answers + per-question rubric (graders only) |
| [`loop_reports/`](loop_reports/) | One report per round: scores, per-question error review (檢討), and the curriculum/exam fixes each round triggered |

## Exam coverage map

Every question is answerable from a specific lesson — the exam tests the course, not trivia:

| Section | Points | Lessons under test |
|---------|--------|--------------------|
| 1. Tensors, autograd & the training loop | 25 | Days 22, 23, 23b (+ Day 33 concepts) |
| 2. Building & feeding models | 25 | Days 24, 25, 25b, 25c, 25d |
| 3. From training to the car: export, ExecuTorch, quantization | 25 | Days 26, 27, 28 |
| 4. Applied: an autonomous-driving inference project | 25 | Days 28b, 28c, 24b, capstone READMEs |

## Re-running the loop

Any capable LLM harness works. For each student: give it the persona above, the curriculum
files, and `final_exam.md` (never the key). Grade with a separate agent holding
`answer_key.md`. Log the round in `loop_reports/roundN.md`, classify every lost point A/B/C,
patch the curriculum (A) or the exam (B), and re-run until all three clear 90.
