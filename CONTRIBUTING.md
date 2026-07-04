# Contributing & Authoring Lessons

This curriculum grows one day at a time. Whether you're the maintainer filling in the
syllabus or a learner who wants to improve a lesson, here's how it works.

## Anatomy of a day

Every `dayNN_topic/` folder follows the same shape (see
[`templates/lesson_template/`](templates/lesson_template)):

```
dayNN_topic/
├── README.md            # goals, the read, how to run
├── lesson.ipynb         # concept + visualizations (optional for pure-coding days)
├── homework.py          # starter code with TODOs
└── test_homework.py     # the auto-grader
```

Design rules we hold to:

1. **Show, don't just tell.** If a concept can be a picture (a growing array, a Gantt chart,
   a loss curve), it should be. Concept notebooks must produce at least one visualization.
2. **The grader is the source of truth.** A learner should be able to work purely against
   `pytest` and know they're done when it's green. Tests must be deterministic and not
   depend on the network.
3. **Every day connects forward.** Say one concrete sentence about where today's idea shows
   up later in the curriculum or in openpilot.
4. **Keep dependencies light.** Only Week 4 should need `torch`. If a test needs an optional
   heavy dep, guard it with `pytest.importorskip("torch")` so the rest of the suite stays green.
5. **~2–3 hours of work, max.** If it's bigger, split it into two days.

## Authoring the concept notebook

We author notebooks from plain Python so they diff cleanly in git and are easy to review.
`tools/nbbuild.py` gives you `md(...)` and `code(...)` helpers; `tools/build_lessons.py`
shows the full pattern for the three sample lessons.

```python
from tools.nbbuild import md, code, write_notebook

write_notebook("curriculum/.../lesson.ipynb", [
    md("# Day NN — Topic", "", "Today you'll learn ..."),
    code("import numpy as np", "import matplotlib.pyplot as plt"),
    md("## The idea", "..."),
    code("# a runnable, visual demo",
         "plt.plot(xs, ys); plt.title('...'); plt.show()"),
])
```

Regenerate all sample notebooks with `make notebooks`.

## Writing the auto-grader

- Import the learner's code from `homework.py`.
- Test **behavior and edge cases**, not implementation details.
- Where a day is about *cost* (e.g. amortized O(1)), assert on the cost too — count resizes,
  comparisons, or operations, don't just check the return value.
- Name tests so a failure reads like a hint: `test_pop_from_empty_raises`, not `test_3`.

## Submitting

1. Copy `templates/lesson_template/` to the right `weekN_*/dayNN_topic/`.
2. Fill in the four files. Run `pytest curriculum/.../dayNN_topic -q` until green.
3. Flip the day's status to ✅ in `SYLLABUS.md` and add it to the "What's built so far"
   list in `README.md`.
4. Open a PR.

> **Maintaining or extending the repo with Claude Code?** There's a ready-to-paste onboarding
> prompt for a fresh session in [`docs/HANDOFF.md`](docs/HANDOFF.md). The prompt below is the
> *student tutor* one; that one is for *maintainers*.

## Suggested Claude Code tutor prompt

Drop this into a `CLAUDE.md` at the repo root, or paste it when you start a session, to turn
Claude Code into a Socratic tutor instead of an answer key:

> You are my patient CS tutor for the learn-openpilot curriculum. When I'm on a lesson:
> explain concepts with concrete analogies and always tie them to *why* they matter for
> systems/robotics code. When my `pytest` fails, help me find the bug by asking guiding
> questions and pointing at the right line — **do not write the solution for me** unless I
> explicitly say "just show me." After I pass a day, quiz me with 3 quick questions and
> suggest one bonus challenge. Keep me connected to the end goal: contributing to openpilot.

Learners: try telling Claude Code *"be stricter — make me struggle a bit more"* or
*"I'm stuck, give me a bigger hint"* to tune how much help you get.
