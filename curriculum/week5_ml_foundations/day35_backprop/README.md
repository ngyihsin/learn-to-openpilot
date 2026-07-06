# Day 35 — Neural Networks & Backpropagation

> **Week 5 · ML & DL Foundations** — the payoff day. You stack layers into a real network and train
> it with **backpropagation**, the algorithm that powers every deep model ever built.
>
> *Assumes basic Python + Days 31–34 (numpy, loss, gradient descent).*

## Why today matters

A single line (Day 33) can only fit straight relationships. Stack a **hidden layer** with a
non-linear activation (**ReLU**) in between, and the network can bend to fit almost any shape — that's
what makes deep learning powerful. But more layers means more knobs, and we can't derive each
gradient by hand like we did for the line.

**Backpropagation** solves this. It's just the **chain rule** applied backwards: compute the loss at
the output, then push the gradient back through each layer — output → ReLU → hidden — reusing the work
at every step. Every framework (PyTorch's autograd on Day 22, and openpilot's training) is doing
exactly what you'll write today, only automatically. Do it once by hand and autograd stops being magic.

### Matrix multiply in 60 seconds (Day 31, Step 8 recap)

Every formula today uses `@` and `.T`. The rules: `(m,k) @ (k,n) → (m,n)` (inner sizes must match),
each output cell = a row of the left dotted with a column of the right, and `.T` swaps rows/columns to
make shapes line up. Worked example: `[[1,2],[3,4]] @ [[5],[6]] = [[1·5+2·6],[3·5+4·6]] = [[17],[39]]`.
If you get a "shapes not aligned" error today, check the inner sizes first — that's always the bug.

### The chain rule in plain words (no calculus needed)

Day 33's gradient said "how does the loss change if I nudge this knob?" With layers, a knob in layer 1
affects the loss only *through* layer 2 — like gears: turning gear 1 turns gear 2, which moves the
needle. The **chain rule** just says: *multiply the effects along the path*. Backprop is bookkeeping
for that — start at the loss, and pass "how much did you affect the error" backwards one layer at a
time, reusing each result. The gradient formulas in the docstrings are exactly this multiplication,
written in matrix form. As on Day 33, you implement them on trust — and the grader's
finite-difference check *proves* they're right by wiggling every parameter and measuring.

### Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **Layer** | A matrix multiply plus a bias: `X @ W + b`. Stacks turn simple parts into a flexible whole. |
| **Hidden layer** | A layer between input and output whose values you never see directly. |
| **ReLU** | `max(0, x)` — the non-linearity that lets stacked layers fit curves, not just lines. |
| **Forward pass** | Run input → output, remembering the in-between values (`z1`, `h`). |
| **Backprop** | Run gradients output → input using the chain rule, producing `dW1, db1, dW2, db2`. |
| **Gradient check** | Confirm your hand-derived gradients match a finite-difference estimate. The test does this. |

### The network you're building

```
X ──(W1, b1)──▶ z1 ──ReLU──▶ h ──(W2, b2)──▶ y ,   loss = MSE(y, target)
   Din             H (hidden)        Dout
```

The chain rule says: to nudge `W1`, you need how the loss changes with `y`, passed back through `W2`,
through the ReLU, to `W1`. That "passing back" is the four lines of `backward()`.

## Learning goals

By the end you can:

- Write a **forward pass** for a 2-layer net and keep the intermediates backprop needs.
- Derive and implement **backprop** — and *prove it's right* with a numerical gradient check.
- Train the whole network with gradient descent and watch the loss fall.

## Do this — six steps

Work top-to-bottom in `homework.py`. The formulas you need are written in the docstrings; your job is
to turn them into numpy and understand *why* each line is there.

1. **`relu(z)`** — `np.maximum(0.0, z)`.
2. **`relu_grad(z)`** — `(z > 0)` as floats.
3. **`forward(X, W1, b1, W2, b2)`** — three lines; return `(y, z1, h)`.
4. **`mse_loss(y_pred, y_true)`** — mean of squared errors (same as Day 33, batched).
5. **`backward(...)`** — the four gradient lines from the docstring, in order.
6. **`train(...)`** — copy the params, then loop: forward → backward → `param -= lr * grad`.

**Check yourself as you go** — one check per stage:

*Forward* (steps 1–3; should print `[[3.]]` — the two hidden units 1 and 2 summed):
```bash
python3 -c "import numpy as np; from homework import forward as f; print(f(np.array([[1.,2.]]), np.eye(2), np.zeros(2), np.array([[1.],[1.]]), np.array([0.]))[0])"
```

*Backward* (step 5; same tiny network, target 1 → error 2, so `db2=[4.]` and `dW2=[4. 8.]`):
```bash
python3 -c "import numpy as np; from homework import forward, backward; X=np.array([[1.,2.]]); y,z1,h=forward(X,np.eye(2),np.zeros(2),np.array([[1.],[1.]]),np.array([0.])); dW1,db1,dW2,db2=backward(X,z1,h,np.array([[1.],[1.]]),y,np.array([[1.]])); print(db2, dW2.ravel())"
# expect: [4.] [4. 8.]
```

*Train* (step 6; the loss must fall — should print `True`):
```bash
python3 -c "import numpy as np; from homework import forward, mse_loss, train; X=np.array([[1.],[2.]]); T=np.array([[2.],[4.]]); W1=np.array([[0.5,0.5]]); b1=np.zeros(2); W2=np.array([[0.5],[0.5]]); b2=np.zeros(1); before=mse_loss(forward(X,W1,b1,W2,b2)[0],T); W1n,b1n,W2n,b2n=train(X,T,W1,b1,W2,b2,lr=0.05,epochs=200); print(mse_loss(forward(X,W1n,b1n,W2n,b2n)[0],T) < before)"
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 35`.
The decisive test is `test_backward_matches_numerical_gradient` — it wiggles every parameter and
confirms your gradients match reality (the finite-difference check from the preamble). If it passes,
your backprop is mathematically correct.

## Check yourself

- Why do we save `z1` and `h` during the forward pass instead of recomputing them in `backward`?
- ReLU's gradient is 0 for negative inputs. What does that mean for a hidden unit that's always
  negative — does it ever learn? (This is the famous "dead ReLU.")
- Backprop reuses `dy` and `dh` on the way back instead of recomputing from scratch. If a network had
  50 layers, why does that reuse matter enormously?

## Where this shows up later

Day 36 swaps this fully-connected block for **self-attention** — a different layer, but trained by the
exact same forward/backward/step loop. From here on you'll let PyTorch's autograd do the `backward()`
for you — but now you know precisely what it computes.

**Next:** Day 36 — Self-attention & Transformers.
