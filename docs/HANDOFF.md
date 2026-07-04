# Handoff — Onboarding a New Claude Code Session

This is a ready-to-paste prompt for spinning up a **fresh Claude Code session** (one with no
memory of this repo) to maintain or extend the curriculum. It front-loads everything a new
instance needs: where things live, the lesson conventions, how to verify, and the environment
gotchas.

**How to use it:** copy the fenced block below into a new Claude Code session and fill in the
`TASK:` line at the top with what you want done. Everything under `TASK:` is standing context —
leave it as-is.

> Looking for the *student tutor* prompt instead (turn Claude into a Socratic coach for someone
> working through the lessons)? That one is in [`CONTRIBUTING.md`](../CONTRIBUTING.md) under
> "Suggested Claude Code tutor prompt." This file is for **maintaining/extending** the repo.

---

````text
You're taking over an existing project: a 30-day interactive CS curriculum called
"learn-openpilot" that teaches data structures → OS → systems programming →
PyTorch/ExecuTorch and culminates in contributing to the openpilot project.

TASK: <describe what you want done — e.g. "add a Week-5 track on C++",
"replace Day 24's synthetic images with a real dataset", "wire up GitHub Actions
CI to run the graders", "review Day 12 for race conditions", or "fix <bug>">

═══════════════════════ WHERE IT LIVES ═══════════════════════
- GitHub: https://github.com/ngyihsin/learn-to-openpilot  (branch: main)
- Clone it fresh, or work in the existing local checkout if present.
- Push finished work to main with clear commit messages.

═══════════════════════ REPO LAYOUT ═══════════════════════
  README.md            vision + quickstart + what's built
  SYLLABUS.md          the full 30-day plan (day → topic → format → status)
  CONTRIBUTING.md      how to author a lesson + a Claude Code "tutor" prompt
  pytest.ini           note: addopts includes --import-mode=importlib
  Makefile             make test / make grade D=<n> / make notebooks
  tools/
    grade.py           progress tracker: `python tools/grade.py status` / `... day <N>`
    nbbuild.py         md()/code()/write_notebook() helpers for authoring .ipynb
    build_lessons.py   ALL lesson notebooks are generated here; run it to (re)emit them
  templates/lesson_template/   copy this to start a new day
  curriculum/
    week1_data_structures/ week2_operating_systems/
    week3_systems_programming/ week4_pytorch_executorch/ capstone_openpilot/
      dayNN_topic/
        README.md          goals, hints, "check yourself", forward links
        homework.py        student STARTER: TODOs that `raise NotImplementedError`
        solution.py        reference solution (SELF-CONTAINED — see rules)
        test_homework.py   the auto-grader
        lesson.ipynb       concept/visualization notebook (generated, not hand-edited)

═══════════════════════ LESSON CONVENTIONS (follow these exactly) ═══════════════════════
1. Every coding day = 4 files: README.md + homework.py (starter) + solution.py
   (reference) + test_homework.py (grader). Concept days also get a lesson.ipynb.
   A few days are guided READMEs only (Day 29, 30) — no grader.
2. test_homework.py loads the implementation via importlib under a UNIQUE module
   name and honors the LP_IMPL env var: default loads `homework`, `LP_IMPL=solution`
   loads `solution`. Copy the exact `_load(...)` helper from any existing
   test_homework.py — it prevents module-name collisions when the whole suite runs.
3. solution.py MUST be self-contained (re-declare any shared dataclasses/types
   inline; do NOT `from homework import ...`). This keeps each folder copyable and
   avoids collisions in the full-suite run.
4. Graders check BEHAVIOR + edge cases + cost (e.g. amortized O(1), Belady's
   anomaly, a time budget), not just happy-path output. Keep them DETERMINISTIC
   (seed torch/rng; join threads; assert on multisets/exact counts). No network.
5. Starters must be red-until-solved (TODOs raise NotImplementedError). It's fine
   if a couple of trivial constructor-guard tests pass on the bare starter.
6. Notebooks are authored in tools/build_lessons.py (as Python via md()/code()),
   NOT hand-edited. Add the lesson's cell list + a targets[] entry, then run
   `python tools/build_lessons.py`. Keep notebook code standalone (it runs on the
   student's machine) and produce at least one visualization per concept day.
7. When you add/finish a day, flip its status to ✅ in SYLLABUS.md, the week's
   README.md, and the main README "what's built" list.

═══════════════════════ HOW TO VERIFY (do this before every push) ═══════════════════════
- Reference solutions must be all-green:   LP_IMPL=solution pytest curriculum
  (currently 205/205 pass across 28 coding lessons; keep it green.)
- The default student run should FAIL (starters unsolved) with NO collection errors:
  pytest curriculum
- Notebooks must be valid JSON after regenerating:
  python tools/build_lessons.py
- Progress map:  python tools/grade.py status

═══════════════════════ ENVIRONMENT GOTCHAS (important) ═══════════════════════
- Install torch from the DEFAULT PyPI index: `pip install torch`.
  The pytorch CDN (download.pytorch.org) is BLOCKED by the proxy — the
  `--index-url .../whl/cpu` route 403s here. pypi.org/files.pythonhosted.org work.
- gcc/cc/clang ARE available (Days 16–17 compile real C; those graders skip if no
  compiler is found).
- The `executorch` package is NOT installable here, so Day 27 uses `torch.export`
  as the lowering step (its grader skips if torch lacks torch.export). Don't try to
  install executorch.
- torch.ao.quantization emits a deprecation warning (Day 28) — harmless, ignore.
- Long commands (full pytest, heavy installs) → run in the background.

═══════════════════════ CURRENT STATE ═══════════════════════
All 30 days are built and pushed; 205/205 grader tests pass against the reference
solutions. Weeks 1–4 complete + capstone Days 29–30. Start by reading README.md and
SYLLABUS.md, then `LP_IMPL=solution pytest curriculum` to confirm green, before
making changes.
````
