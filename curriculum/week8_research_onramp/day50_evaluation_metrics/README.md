# Day 50 — Evaluating Models: Metrics Done Right

> **Week 8 · Research On-ramp** — a result is only as trustworthy as the metric behind it. "Accuracy"
> lies on imbalanced data; **precision** and **recall** tell the truth. Build them, then you can judge
> any paper's numbers — and your own.
>
> *Gradable: assumes basic Python + Day 31 numpy.*

## Why today matters

Imagine a pedestrian detector on a street where 99% of frames have no pedestrian. A model that *always*
says "no pedestrian" scores **99% accuracy** — and misses every actual person. Accuracy hid a fatal
failure. Research uses metrics that don't:

- **Precision** — when the model says "positive," how often is it right? (Low precision = many false alarms.)
- **Recall** — of all the real positives, how many did it catch? (Low recall = dangerous misses.)
- **F1** — a single number balancing the two, for when you need one score.

These come from four counts (**TP, FP, FN, TN**). Compute those and everything else follows.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **TP / FP / FN / TN** | true-positive, false-positive (false alarm), false-negative (miss), true-negative. |
| **precision** | `TP / (TP + FP)` — trustworthiness of a positive prediction. |
| **recall** | `TP / (TP + FN)` — coverage of the real positives. |
| **F1** | `2·P·R / (P + R)` — harmonic mean; low if *either* is low. |
| **the accuracy trap** | On imbalanced data, high accuracy can coexist with zero recall. |

## See it with real numbers

Truth `[1,1,0,0,1]`, predictions `[1,0,0,1,1]`:
TP=2 (caught 2 of 3 positives), FP=1 (one false alarm), FN=1 (one miss), TN=1.
So precision = 2/3, recall = 2/3, F1 = 2/3.

## Do this — four steps

Work top-to-bottom in `homework.py`; run each check in this folder.

**Step 1 · `confusion(y_true, y_pred)`** — count TP, FP, FN, TN with boolean masks.
```bash
python3 -c "from homework import confusion as f; print(f([1,1,0,0,1],[1,0,0,1,1]))"   # expect: (2, 1, 1, 1)
```

**Step 2 · `precision(tp, fp)`** — `tp / (tp + fp)`, or 0.0 if none predicted positive.
```bash
python3 -c "from homework import precision as f; print(round(f(2,1),4))"              # expect: 0.6667
```

**Step 3 · `recall(tp, fn)`** — `tp / (tp + fn)`, or 0.0 if no real positives.
```bash
python3 -c "from homework import recall as f; print(round(f(2,1),4))"                 # expect: 0.6667
```

**Step 4 · `f1_score(precision, recall)`** — `2*p*r / (p + r)`, or 0.0 if both are 0.
```bash
python3 -c "from homework import f1_score as f; print(round(f(2/3, 2/3),4))"          # expect: 0.6667
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 50`.

## Check yourself

- A model has 99% accuracy but 0% recall on a rare class. What did accuracy hide, and which metric
  exposed it?
- A spam filter with high **recall** but low **precision** — what annoying behavior does that describe?
  What about high precision, low recall?
- Why the *harmonic* mean for F1 instead of a plain average? (What happens to F1 if precision is 0.99
  but recall is 0.01?)

## Where this shows up later

Day 51 uses these to compare experiments fairly (did my change actually help?). In detection (Week 7), a
prediction "counts" as correct when its IoU (Day 42) with a ground-truth box clears a threshold — then
you compute exactly these metrics on the matches. Honest evaluation is what separates a result from a
guess.

**Next:** Day 51 — Experiments, tracking & ablations.
