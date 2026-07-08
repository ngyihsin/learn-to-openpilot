# Week 6 — The Research Toolchain

> *Goal: everything around the model. You can train a net now (Week 5) — this week you learn to run
> other people's code, on real hardware, reproducibly. This is the advisor's **Stage 1**: "be able to
> reproduce paper code," with the required tooling — Linux, Git, Docker, CUDA.*

Weeks 1–5 were about *writing* code. Research is mostly about *running* code someone else wrote — a
paper's GitHub repo, on a GPU box, and getting the *same* numbers they did. That takes a different
skill set, and it's exactly what your future lab expects on day one.

| Day | Topic | Format | You'll be able to | Status |
|-----|-------|--------|-------------------|--------|
| 37 | **Linux, the shell & your GPU environment** | guided | Navigate the shell, make a venv, check your CPU/GPU/CUDA | ✅ |
| 38 | **Git & version control for research** | guided | Branch, commit, `.gitignore` big files, read a repo's history | ✅ |
| 39 | **Docker & reproducible environments** | guided | Understand images/containers; run a project in a container | ✅ |
| 40 | **Reproducing a paper end-to-end** | guided | Clone → env → weights → run inference → match the reported result | ✅ |
| 41 | **Structuring a PyTorch research project** | pytest | Configs, seeds, logging, checkpoints — build reproducibility utils | ✅ |

**Note on format:** Days 37–40 are **guided, hands-on READMEs** — you *run* commands and check the
output, rather than filling in `homework.py`. They have no auto-grader (you can't unit-test "did Docker
install"), the same style as the openpilot on-ramp (Days 29–30). Day 41 comes back to gradable code:
the reusable bits (seeding, config loading) *can* be tested, and they're what make experiments trustworthy.

**Why now:** Week 7 has you cloning YOLO / SAM / Grounding DINO and running them on KITTI/NuScenes.
None of that works until you're fluent with the shell, environments, and Git — so we build that first.

**Next:** Week 7 — Computer Vision & Vision-Language Models.
