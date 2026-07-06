# Day 32 — The ML Framing: Model, Loss, Generalization

> **Week 5 · ML & DL Foundations** — before you train anything, understand what "training" even
> means. This is the vocabulary the entire rest of the course is spoken in.
>
> *Assumes you can write basic Python (variables, functions, lists, loops). Everything about numpy
> and machine learning is explained here.*

## Why today matters

Every ML system — a line-fitter or openpilot's driving model — is the same three parts:

1. a **model**: a function with adjustable knobs (its *parameters*),
2. a **loss**: one number saying how wrong the model is on some data,
3. a search: nudge the knobs until the loss is small.

But small loss on the data you *trained on* is not the goal. The goal is **generalization** — doing
well on data the model has **never seen**. A model can score perfectly on the training data by simply
memorizing it, yet be useless on anything new. That failure is **overfitting**, and catching it is
most of the job. Today you build the smallest honest version of the whole idea.

### Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **Model** | A function whose output you can change by turning knobs (parameters). Today: a polynomial. |
| **Loss** | A single "how wrong are we" number. Lower is better. Today: **MSE** (below). |
| **MSE** | *Mean Squared Error* — average of the squared misses. Squaring makes big misses hurt more and treats −3 and +3 the same. |
| **Training set** | The data you're allowed to fit the knobs on. |
| **Validation set** | Data you hold back and *never* fit on — used only to check if the model generalizes. |
| **Overfitting** | Low training loss but high validation loss: the model memorized instead of learning. |

### See it with real numbers first

**MSE:** true answers `[1, 2]`, your guesses `[3, 4]`:
misses `2` and `2` → squared `4` and `4` → average `(4+4)/2 = ` **`4.0`**. That's the whole loss.

**Train/validation split:** 10 rows, `val_frac=0.2` → hold out `round(10 × 0.2) = 2` rows for
validation, keep 8 for training. The 2 held-out rows are the "exam questions" the model never studies.

### Watch overfitting happen

Fit polynomials of rising degree to the *same* noisy data (from a degree-2 truth) and measure error on
both sets:

```
 degree | train MSE | val MSE
    1    |  0.23209  | 0.18608
    2    |  0.00142  | 0.00371   ← validation bottoms out here
    5    |  0.00117  | 0.00412
    8    |  0.00083  | 0.00388
   12    |  0.00077  | 0.00422   ← train still falling, val creeping UP
```

Training error keeps dropping as the model gets more complex — but validation error turns around at
degree 2. That U-turn **is** overfitting: past degree 2 the model is memorizing noise, not learning the
pattern. `lesson.ipynb` plots this curve and draws the wiggly degree-15 fit so you can see it chase the
noise. Your `select_degree` is just "pick the bottom of the validation curve."

## Learning goals

By the end you can:

- Compute the **MSE** loss in code, and say why we square the errors.
- Split data into **train** / **validation** correctly, keeping each label with its features.
- Fit models of different complexity and **pick the one with the lowest *validation* error** — and
  explain why that's different from lowest *training* error.

## Do this — five small steps

0. **Concept + visualization (~10 min).** Open `lesson.ipynb` and run every cell to *see* the
   overfitting curve above and what each model looks like — then come back and build it.

Work top-to-bottom in `homework.py`. After each function, run its one-line check in this folder and
confirm the output matches. (These are the same ideas the grader tests — passing them means you're on track.)

**Step 1 · `mse(y_true, y_pred)` — the loss.**
Turn both inputs into float arrays, subtract, square, take the mean. In numpy that's
`np.mean((a - b) ** 2)` — no loop needed; math applies to whole arrays at once.
```bash
python3 -c "from homework import mse; print(mse([1,2],[3,4]))"        # expect: 4.0
```

**Step 2 · `predict_polynomial(coeffs, x)` — run the model.**
`np.polyval(coeffs, x)` evaluates a polynomial. Coefficients are **highest power first**, so
`[2,-3,1]` means `2x² − 3x + 1`.
```bash
python3 -c "from homework import predict_polynomial as p; print(p([2,-3,1], 0))"   # expect: 1.0
```

**Step 3 · `fit_polynomial(x, y, degree)` — turn the knobs.**
`np.polyfit(x, y, degree)` finds the best coefficients for you. Fit a degree-2 curve to points that
came from `2x² − 3x + 1` and you should recover those coefficients:
```bash
python3 -c "import numpy as np; from homework import fit_polynomial as f; x=np.linspace(-3,3,50); print(np.round(f(x, 2*x**2-3*x+1, 2), 2))"   # expect: [ 2. -3.  1.]
```

**Step 4 · `train_val_split(X, y, val_frac, seed)` — hold out an exam set.**
Make a reproducible shuffle of the *row positions*, then slice:
`idx = np.random.default_rng(seed).permutation(len(X))`. Take the first `round(len(X)*val_frac)`
positions as validation, the rest as train, and index **both** `X` and `y` with them (so each label
stays with its features — `X[idx]` reorders an array by a list of positions).
```bash
python3 -c "import numpy as np; from homework import train_val_split as s; print([len(a) for a in s(np.arange(10), np.arange(10), 0.2, 0)])"   # expect: [8, 8, 2, 2]
```

**Step 5 · `select_degree(...)` — pick the model that generalizes.**
Loop over the candidate degrees. For each: `fit_polynomial` on the **train** data, `predict` on the
**validation** data, score with `mse`. Return the degree with the **smallest validation MSE**. A very
high degree will fit the training noise (low train loss) but score *worse* on validation — the loop
rejects it for you.
```bash
python3 -c "import numpy as np; from homework import select_degree as f; x=np.linspace(-2,2,30); v=np.linspace(-1.8,1.8,15); print(f(x, x**2, v, v**2, [0,1,2]))"   # expect: 2
```
*(The data is a pure parabola, so degree 2 must win — degrees 0 and 1 can't bend to fit it.)*

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 32`.

## Check yourself

- A degree-15 fit has *lower training error* than degree-2 on the same noisy data. Why is it still the
  worse model, and what does its *validation* error do?
- Why must you split off validation **before** fitting, and never touch it while choosing knobs?
- If you only ever looked at training error, how could you be fooled into shipping a memorizer?

## Where this shows up later

Day 33 throws away `np.polyfit`'s closed-form magic and finds the knobs by **gradient descent** — the
search that scales to millions of parameters. The train/validation discipline you built today is how
every experiment in Week 8 and every model in openpilot is judged.

**Next:** Day 33 — Regression & gradient descent.
