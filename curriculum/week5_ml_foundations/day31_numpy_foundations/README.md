# Day 31 — numpy & Array Thinking

> **Week 5 · ML & DL Foundations** — the first day of the machine-learning half. Before we learn
> *what* a model is, we learn the tool every model is built and run with: **numpy**.
>
> *Assumes basic Python (variables, functions, lists, loops). Everything about numpy is taught here.*

## Why today matters

From here on, data isn't a Python list — it's a numpy **array**: an image is an array of pixels, a
KITTI/NuScenes scan is an array of points, a model's input and output are arrays. Every paper you'll
reproduce for your research speaks numpy.

The one idea that makes numpy click is **vectorization**: math applies to the *whole array at once*,
with no `for` loop. This isn't just tidier — it's often 10–100× faster, because the loop happens in
optimized C under the hood instead of in Python. Learn to *think in arrays* and the rest of the
course reads naturally.

### The one gotcha to see first

A Python list and a numpy array look similar but behave differently under math:

```python
[1, 2, 3] * 2                      # -> [1, 2, 3, 1, 2, 3]   (list: repeats!)
np.array([1, 2, 3]) * 2            # -> [2 4 6]              (array: doubles each)
```

That second behavior — arithmetic touching every element — is the whole point.

### Key ideas (plain language)

| Idea | In one sentence |
|------|-----------------|
| **array** | A grid of numbers, all the same type. `np.array([1,2,3])`. |
| **shape** | The size along each dimension. `(3,)` is 3 numbers; `(2,3)` is 2 rows × 3 cols. |
| **vectorization** | `a + b`, `a * 2`, `np.abs(a)` act on every element — no loop. |
| **broadcasting** | A scalar (or smaller array) auto-stretches to fit: `x - x.mean()` subtracts one number from all. |
| **boolean mask** | `x > 0` gives an array of True/False; `x[x > 0]` keeps only the True ones. |
| **fancy indexing** | Index by a *list of positions*: `x[[2,0,1]]` picks those elements (and can reorder/shuffle). |
| **axis** | Which dimension a reduction collapses. `M.mean(axis=0)` = per-column; `M.mean(axis=1)` = per-row. |
| **matmul `@`** | Matrix multiply: `(m,k) @ (k,n) → (m,n)` — inner sizes must match. Each output cell = row of A · column of B. |
| **transpose `.T`** | Swap rows and columns: a `(2,3)` array becomes `(3,2)`. Used to make shapes line up for `@`. |
| **rng** | A *seeded* random generator: `np.random.default_rng(seed)` — same seed, same numbers, every run. |

## Learning goals

By the end you can:

- Do elementwise math and **broadcasting** on arrays instead of writing loops.
- Filter an array with a **boolean mask**, and reduce along a chosen **axis**.
- Use `argmin` to find a nearest value, and draw **reproducible** random numbers from a seed.

## Do this — eight small steps

0. **Concept + visualization (~10 min).** Open `lesson.ipynb` and run every cell — you'll *see*
   numpy beat a Python loop on speed, watch broadcasting stretch one number across an array, and
   compare mask-indexing vs position-indexing before you write any code.
1. Work top-to-bottom in `homework.py`. After each function, run its check in this folder and confirm
   the output. (`np.` prints arrays without commas, e.g. `[1. 3. 5.]` — that's normal.)

**Step 1 · `scale_and_shift(x, a, b)` — elementwise math.** Return `a * x + b` over the whole array.
```bash
python3 -c "from homework import scale_and_shift as f; print(f([0,1,2], 2, 1))"      # expect: [1. 3. 5.]
```

**Step 2 · `normalize(x)` — broadcasting + aggregation.** Return `(x - x.mean()) / x.std()`. One
number (the mean) is subtracted from every element — that's broadcasting.
```bash
python3 -c "from homework import normalize as f; print(f([1,2,3]))"                  # expect: [-1.22474487  0.          1.22474487]
```
*(Where does −1.22 come from? The mean of `[1,2,3]` is 2 and the std is ≈0.816, so `(1 − 2) / 0.816 ≈ −1.22`. Normalizing re-centers data to mean 0 and rescales it to std 1.)*

**Step 3 · `select_positive(x)` — boolean mask.** Keep only elements `> 0` via `x[x > 0]`.
```bash
python3 -c "from homework import select_positive as f; print(f([-1,2,-3,4,0]))"      # expect: [2. 4.]
```

**Step 4 · `row_means(M)` — reduce along an axis.** Mean of each *row* of a 2-D array: `M.mean(axis=1)`.
```bash
python3 -c "from homework import row_means as f; print(f([[1,2,3],[4,5,6]]))"        # expect: [2. 5.]
```

**Step 5 · `closest_index(x, value)` — argmin.** Index of the element nearest `value`:
`np.argmin(np.abs(x - value))`. (This is the trick behind "which class is closest" and Day 33's model
selection.)
```bash
python3 -c "from homework import closest_index as f; print(f([10,20,30], 22))"       # expect: 1
```

**Step 6 · `reproducible_randoms(n, seed)` — seeded randomness.** `np.random.default_rng(seed).random(n)`.
Same seed ⇒ same numbers, which is how experiments stay reproducible.
```bash
python3 -c "from homework import reproducible_randoms as f; print(f(3, 0))"          # expect: [0.63696169 0.26978671 0.04097352]
```

**Step 7 · `reorder(values, order)` — fancy (position) indexing.** Where a mask selects by a
True/False test, `values[order]` selects by a *list of positions* — the trick Day 32 uses to shuffle
data. `values[[2, 0, 4]]` returns elements 2, 0, 4 in that order.
```bash
python3 -c "from homework import reorder as f; print(f([5,-2,9,-1,4], [2,0,4]))"    # expect: [9. 5. 4.]
```

**Step 8 · `matmul(A, B)` — matrix multiply.** The one operation deep learning is made of: `A @ B`.
Worked by hand: `[[1,2],[3,4]] @ [[5],[6]]` — each output cell is *a row of A times a column of B,
summed*: row 1 gives `1·5 + 2·6 = 17`, row 2 gives `3·5 + 4·6 = 39`. The shape rule: `(m,k) @ (k,n)`
→ `(m,n)`, so the **inner sizes must match** — and when they don't, `.T` (transpose: swap
rows/columns) usually fixes it. Days 35–36 use `@` and `.T` in every formula.
```bash
python3 -c "from homework import matmul as f; print(f([[1,2],[3,4]], [[5],[6]]))"   # expect: [[17.] [39.]]
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 31`.

## Check yourself

- Why is `np.array([1,2,3]) * 2` different from `[1,2,3] * 2`? Which one does a model need?
- In `x - x.mean()`, `x` has 3 numbers but `x.mean()` is 1 number. How does the subtraction work?
- For a 2-D array, what does `axis=0` collapse vs `axis=1`? (Rows vs columns — which is which?)
- Why do real experiments *seed* their randomness instead of using truly random numbers?

## Where this shows up later

Day 32 uses these exact operations to compute a loss and split data; Day 33 turns `scale_and_shift`
into a trained linear model. Every later week — CNNs, detection, loading NuScenes — is arrays all the
way down.

**Next:** Day 32 — The ML framing: model, loss, generalization.
