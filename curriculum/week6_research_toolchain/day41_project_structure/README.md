# Day 41 — Structuring a PyTorch Research Project

> **Week 6 · Research Toolchain** — the difference between an experiment you can trust and one you
> can't is a handful of small habits. Today you build them as reusable utilities.
>
> *Back to gradable code: `assumes basic Python + Day 31 numpy`.*

## Why today matters

A research result is only real if you (or a reviewer) can **rerun it and get the same number**. That
takes four unglamorous habits, and every serious repo has them:

- **Seed everything** — so "random" is repeatable.
- **Track metrics cleanly** — a running average, not a pile of loose variables.
- **Manage config** — sensible defaults you can override per experiment, without editing code.
- **Pick the best checkpoint** — by *validation* metric (the Day 32 lesson, applied to training runs).

You'll build a small, tested version of each. These are exactly the helpers you'll paste into your
master's project.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **seed** | A number that makes a random sequence repeatable. Same seed ⇒ same run. |
| **config** | The knobs of an experiment (lr, epochs, model) kept as data, not hard-coded. |
| **checkpoint** | A saved model state; you keep the *best* one by validation metric. |
| **meter** | A tiny accumulator for a running average of a metric. |

> **New Python here:** `AverageMeter` is a **class** — you'll write `self`, methods, and an
> `@property` (a method read *without* parentheses, like a variable). If those are new or rusty, do
> **Day 00 — The Python this course is written in** first (`curriculum/week0_python_bridge/`); it
> teaches exactly these in one sitting. You'll also `raise ValueError` in `pick_best` — Day 00
> covers that too.

## Do this — four steps

Work top-to-bottom in `homework.py`; run each check in this folder.

**Step 1 · `set_seed(seed)`** — `random.seed(seed); np.random.seed(seed)`.
```bash
python3 -c "import random,numpy as np; from homework import set_seed; set_seed(0); print(round(random.random(),4), round(float(np.random.rand()),4))"
# expect (reproducible): 0.8444 0.5488
```

**Step 2 · `AverageMeter`** — `update(value, n)` accumulates; `avg` returns the mean (0.0 if empty).
```bash
python3 -c "from homework import AverageMeter; m=AverageMeter(); m.update(2); m.update(4); print(m.avg)"   # expect: 3.0
```

**Step 3 · `merge_config(defaults, overrides)`** — a new dict, overrides on top, inputs untouched.
```bash
python3 -c "from homework import merge_config as m; print(m({'lr':0.1,'epochs':10}, {'lr':0.01}))"   # expect: {'lr': 0.01, 'epochs': 10}
```

**Step 4 · `pick_best(history, key, mode)`** — the record with the min (loss) or max (accuracy) value.
```bash
python3 -c "from homework import pick_best as p; print(p([{'epoch':1,'val_loss':0.5},{'epoch':2,'val_loss':0.3}], 'val_loss')['epoch'])"   # expect: 2
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 41`.

## Check yourself

- If you *don't* seed, two runs of the same code give different numbers. Why does that make a bug report
  or a result impossible to trust?
- Why should `merge_config` return a *new* dict instead of modifying `defaults` in place? (What happens
  to the next experiment if you mutate the shared defaults?)
- `pick_best` uses `mode='min'` for loss but `mode='max'` for accuracy. Why can't one rule cover both?

## Where this shows up later

Week 7 runs models whose repos are full of exactly these utilities — you'll recognize them now. Week 8
uses `pick_best` and seeded runs to compare experiments honestly. This is the backbone of a reproducible
research project.

**Next:** Week 7 — Computer Vision & Vision-Language Models.
