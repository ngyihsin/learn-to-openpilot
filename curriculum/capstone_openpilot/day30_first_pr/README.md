# Day 30 — Open Your First Pull Request

> **Capstone finale** — 29 days of fundamentals, now aimed at a real contribution.

Yesterday you cloned, built, and explored openpilot, and scouted a candidate issue. Today you
close the loop: make a focused change, back it with a test, run the project's checks, and open a
**pull request**. This is the day the whole curriculum has been building toward — you're not
studying anymore, you're contributing.

> ⚠️ **Safety first, again.** Everything here happens on your machine against recorded/simulated
> data. Your first PR should **never** touch safety-critical vehicle-control logic. Great first
> contributions are tooling, docs, tests, error messages, and small well-scoped bug fixes.

## Learning goals

By the end you can:

- Turn an issue into a small, reviewable change on a feature branch.
- Write a **test that fails before your fix and passes after** — the heart of trustworthy changes.
- Run a real project's checks (linters, formatters, test suite) and read CI results.
- Open a clean pull request that follows the project's `CONTRIBUTING.md`, and respond to review.

## 0. Re-read the rules (5 min, non-negotiable)

Open openpilot's **`CONTRIBUTING.md`** and skim recent merged PRs. Every project has its own
conventions — commit style, how tests are run, what a reviewer expects, whether a CLA is required.
**Follow that file to the letter; it overrides any generic advice here.** Part of being a good
contributor is respecting the house rules.

## 1. Pick the change and branch

From your Day 29 shortlist, choose the *smallest* real improvement you understand end to end. Then:

```bash
cd openpilot
git checkout -b fix/short-descriptive-name
```

Small and correct beats big and impressive. Reviewers merge changes they can fully understand.

## 2. Reproduce first, then fix

Don't fix what you can't reproduce. Write the failing case *first* — this is Week 3's Day 20
instinct (measure, don't guess) applied to correctness:

1. Find the code that owns the behavior (use Claude Code: *"where is X handled in this repo?"*).
2. Add or find a test that **fails** because of the bug. Run it, watch it fail.
3. Make the minimal change so the test passes — and nothing else breaks.

A change with a test that pins the behavior is far more likely to merge than a bare fix.

## 3. Run the project's checks locally

Before you push, make it pass what CI will run. openpilot uses tooling like `ruff`/linters,
formatters, and `pytest`; many repos wire these into **pre-commit**. Find the current commands in
`CONTRIBUTING.md` / the repo docs — then:

```bash
# examples — use whatever the repo actually specifies:
pre-commit run --all-files       # if the repo uses pre-commit
# run the relevant test file(s), including your new test
pytest path/to/affected/tests -q
```

Green locally means fewer round-trips with reviewers and CI.

## 4. Commit and push to your fork

```bash
git add <the specific files you changed>     # be intentional — no stray files
git commit -m "clear message: what changed and why"
git push -u origin fix/short-descriptive-name
```

Write the commit message for a future reader: *what* and *why*, not just *what*.

## 5. Open the pull request

On GitHub, open a PR from your branch to `commaai/openpilot`'s default branch. A strong PR:

- **Title:** concise and specific.
- **Body:** what problem it solves, how you fixed it, how you verified it (mention the test),
  and a link to the issue (`Fixes #1234` if applicable).
- **Scope:** one focused change. Resist bundling unrelated tweaks.
- Follows any PR template the repo provides.

Then watch **CI**. If it's red, read the logs, reproduce locally, fix, and push again — the same
"kick it until it's green" loop, driven by *you*.

## 6. Respond to review like a pro

- Reply to every comment; make requested changes as new commits (don't force-push away history
  unless asked).
- If you disagree, explain your reasoning politely and with evidence — reviewers are collaborators,
  not adversaries.
- Be patient. Maintainers are busy; a well-scoped, well-tested PR respects their time and gets
  merged faster.

## Check yourself

- Why write the failing test *before* the fix?
- Why is a small, single-purpose PR more likely to merge than a large one?
- If CI fails on something unrelated to your change, what do you do? (Reproduce, ask, don't blindly
  retry forever.)

## You made it — how the 30 days paid off

Everything you built shows up here:

| What you learned | How it served you today |
|---|---|
| Data structures & Big-O (W1) | Reading the code and reasoning about the fix |
| Processes, threads, IPC (W2) | Understanding openpilot's multi-process design |
| C, syscalls, sockets, builds (W3) | Building the project and reading its C++/tooling |
| PyTorch → ExecuTorch, quantization (W4) | Understanding the model that runs on the device |
| TDD & the "get CI green" loop (throughout) | Shipping a change with a test that proves it |

You started as someone learning arrays. You're finishing as someone who can land a pull request on
a real, complex, open-source robotics project. Keep going: pick another issue, go a little deeper,
and let each contribution teach you the next piece. **Welcome to open source.** 🚗💨

---

*openpilot is an independent project of comma.ai. Contribute responsibly, follow their
`CONTRIBUTING.md` and community guidelines, and never test unreviewed code on a real vehicle.*
