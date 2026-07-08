# Day 23b — Tensor Shapes & Broadcasting

> **Week 4 · PyTorch / ExecuTorch** — the debugging skill that unblocks every "size mismatch" error.

## Why today matters

The moment you clone a real repo — YOLO, SAM, a Hugging Face model — and press run, the first
wall you hit is almost never a subtle algorithm bug. It's a shape error:
`RuntimeError: The size of tensor a (3) must match the size of tensor b (4)`. Beginners stare at
that line and stall for an hour. People who can *read a shape and predict a broadcast* fix it in
thirty seconds.

That skill is entirely learnable, and today you learn it. A tensor's **shape** is the single most
important thing to keep in your head while debugging: which axis is the batch, which is channels,
which is height and width. And **broadcasting** — PyTorch's rule for stretching a small tensor to
meet a bigger one — is what lets `x @ W.T + b` add one bias vector across a whole batch without a
loop. Get these two ideas solid and the "size mismatch" wall stops being a wall.

## Learning goals

By the end you can:

- Apply the broadcasting rule by hand: align shapes **from the right**, and each dimension must be
  equal or one of them must be `1`.
- Read a matrix-multiply shape and say why `(N, in) @ (in, out) → (N, out)`.
- Reshape deliberately, and recognize the "numel doesn't match" error before PyTorch throws it.
- Normalize a `(C, H, W)` image per-channel with `keepdim` broadcasting — the exact preprocessing
  step in front of every vision model.

## Do this

1. **Homework (~60 min).** Implement the four functions in `homework.py`:
   - `broadcast_shapes(a, b)` — compute the result shape by hand (don't call
     `torch.broadcast_shapes`; you're learning the rule *is* the point).
   - `batched_linear(x, W, b)` — the fully-connected layer, `x @ W.T + b`.
   - `to_shape(x, shape)` — reshape with a clear error when the element count is wrong.
   - `channel_normalize(img)` — per-channel mean/std normalization with broadcasting.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 23b
   ```
   The grader checks your `broadcast_shapes` against `torch.broadcast_shapes` on tricky cases,
   confirms your linear layer matches `x @ W.T + b`, and checks the normalized image really has
   per-channel mean 0 and std 1.

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- **Broadcasting, said once, precisely:** pad the shorter shape with `1`s on the *left* until both
  have the same length, then compare dimension by dimension. Two dims are compatible if they're
  equal, or either is `1`; the output takes the bigger of the two. If any pair fails, it's an
  error — that's your `ValueError`.
- For `batched_linear`, remember `W` is stored as `(out, in)` (PyTorch's `nn.Linear` convention),
  so you transpose it to `(in, out)` before the matmul. The bias `(out,)` broadcasts across all
  `N` rows for free — that's broadcasting earning its keep.
- `to_shape`: `x.numel()` is the total element count; the target size is the product of `shape`.
  Compare them yourself and raise a message that prints *both* numbers, so the error tells you what
  went wrong instead of making you guess.
- For `channel_normalize`, reduce over the H and W axes with `dim=(1, 2), keepdim=True`. `keepdim`
  keeps the result shaped `(C, 1, 1)` so it broadcasts cleanly back over `(C, H, W)`. Add `eps` to
  the std *before* dividing.

## Check yourself

- Why does `(2, 1, 4)` broadcast with `(5, 4)` but `(3,)` fails against `(4,)`? Walk the
  right-aligned columns out loud.
- In `x @ W.T + b`, what shape is each intermediate? Where exactly does the bias broadcast?
- If you forgot `keepdim=True` in `channel_normalize`, what shape would the mean be, and why would
  the subtraction then either error or broadcast the *wrong* way?

## Where this shows up later

Day 24 stacks convolutions into a CNN, and every layer is a shape transformation you'll now be
able to trace: `(N, 3, H, W) → (N, 64, H/2, W/2) → …`. When you load a pretrained vision or VLM
model later in Weeks 5–7 and it complains about a mismatch, this is the day that lets you read the
error, find the offending axis, and fix it — instead of pasting it into a search box and hoping.

**Next:** Day 24 — CNNs & Computer Vision.
