# Day 40 — Reproducing a Paper End-to-End

> **Week 6 · Research Toolchain** — the capstone of Stage 1, and the single skill your advisor asked
> for first: **take a paper's public code and get its result running on your machine.**
>
> *Guided, hands-on: a repeatable checklist you'll use for every repo in Week 7.*

## Why today matters

Reading a paper tells you *what* was done; reproducing it tells you whether you actually understand it —
and it's how real research starts (you reproduce, then modify). Everything from Week 6 comes together
here: the **shell** to run it, a **venv/Docker** for the environment, **Git** to clone it, and your
**GPU/CPU** check to make it run. The skill isn't any one command — it's a calm, ordered process for
turning a stranger's repo into a working demo.

## The reproduction checklist

Work through these in order. 80% of failures are environment mismatches, not bugs.

**1 · Pick a repo and read the README first.**
```bash
git clone <repo-url>
cd <repo>
# Read README.md end to end BEFORE running anything. Look for: Python version,
# install steps, how to get weights/data, and the exact command to run inference.
```

**2 · Make a clean, isolated environment.** (Never install a paper's deps into your system Python.)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or follow the README's install steps / Dockerfile
```

**3 · Get the weights and a sample input.** Models ship separately from code (too big for Git).
```bash
# The README usually has a download link or a script, e.g.:
#   wget <weights-url> -O weights.pth
#   bash scripts/download_checkpoints.sh
```

**4 · Run inference on ONE sample first.** Smallest possible test before anything ambitious.
```bash
python demo.py --input sample.jpg --weights weights.pth
# device-agnostic: good repos auto-pick cuda/cpu; if it hard-codes cuda and you're on CPU, that's fixable.
```

**5 · Compare to the reported result.** Same input → does your output match the paper's figure/number?
Small differences are normal (hardware, library versions); wildly different means something's off.

## When it breaks (it will) — the usual suspects

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError` | dep not installed / wrong env | activate the venv; `pip install <name>` |
| `CUDA error` / `device` mismatch | code assumes a GPU | set device from `torch.cuda.is_available()`; or run CPU |
| Version error (torch/numpy) | newer libs than the paper used | pin the README's versions in a fresh venv |
| Shapes/keys don't match | wrong weights file | re-download the exact checkpoint the README names |
| Downloads fail | dead link / needs login | check the repo's Issues; find a mirror |

> **Read the repo's GitHub Issues.** Whatever error you hit, someone probably already reported it — and
> the maintainer's answer is often the exact fix. This is the highest-leverage reproduction habit.

## Check yourself

- Why run inference on **one** sample before the full dataset?
- The repo needs `torch==1.13` but you have `2.4`. What's the safe fix — upgrade the repo, or pin the
  version in a fresh venv? Why?
- Your output differs from the paper's by a tiny amount. Reproduction failure, or expected? What would
  a *large* difference suggest instead?

## Where this shows up later

Week 7 *is* this checklist, four times: you'll clone and run **YOLO**, **SAM**, and **Grounding DINO**,
and load **KITTI/NuScenes** data. Come back to this page each time. In your master's work, "I reproduced
the baseline" is the sentence that unlocks a real project.

**Next:** Day 41 — Structuring a PyTorch research project.
