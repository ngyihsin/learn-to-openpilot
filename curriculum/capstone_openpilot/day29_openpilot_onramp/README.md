# Day 29 — The openpilot On-Ramp

> **Capstone** — everything you built in 28 days, pointed at a real, running, open-source self-driving stack.

You've built data structures, reasoned about scheduling and memory, crossed the C/Python
boundary, and trained & exported a model. **openpilot** uses *all* of it at once. Today is a
guided tour: understand the architecture, build it, run it on recorded data, and get oriented
so that tomorrow (Day 30) you can open your first pull request.

> ⚠️ **Safety first.** openpilot is real driving software. Everything here is done on your
> computer against **recorded/simulated data** — you do not need a car or a comma device to
> learn, contribute, and get PRs merged. Never test unreviewed code on a real vehicle.

## Learning goals

By the end you can:

- Describe openpilot's process-based architecture and how data flows camera → model → planner → controls.
- Clone and **build** openpilot, and **replay** a recorded route to see the stack run.
- Navigate the repo well enough to find where a given behavior lives.
- Identify a realistic **good-first-issue** and how you'd verify a fix.

## 1. The mental model (map your 28 days onto it)

openpilot is not one big program — it's **many small processes** communicating over a
pub/sub message bus. That should sound familiar:

| openpilot piece | The day you learned the idea |
|---|---|
| `cereal` / `msgq` — messages between processes via shared memory | Day 08 processes, Day 12 IPC/shared memory |
| Processes running at fixed rates (camerad, modeld, plannerd, controlsd) | Day 08 scheduling, real-time deadlines |
| The driving **model** (vision → path) | Days 22–24 tensors, training, CNNs |
| Model runs **on-device**, quantized for speed | Days 26–28 export, ExecuTorch, quantization |
| C++ core with Python tooling, built with **scons** | Days 15–17 memory & C/Python, Day 21 build systems |

Rough data flow:

```
 camera frames ──► camerad ──► modeld (neural net) ──► plannerd ──► controlsd ──► car
                     │             │                      │            │
                     └──────── cereal / msgq pub-sub message bus ──────┘
                          (logged so any route can be replayed offline)
```

**Do:** skim the top-level `README.md` and the `system/`, `selfdrive/`, and `cereal/`
directories in the openpilot repo. For each, write one sentence: *what runs here?*

## 2. Get the code and build it

Work against your own fork (you'll need it for the PR on Day 30):

```bash
# On GitHub: fork commaai/openpilot to your account, then:
git clone --recurse-submodules https://github.com/<you>/openpilot.git
cd openpilot

# Ubuntu / macOS: the repo ships a setup script that installs dependencies
tools/op.sh setup      # or follow docs/ if this path has changed
tools/op.sh build      # scons under the hood — the build system from Day 21
```

> openpilot's tooling evolves. If a command has moved, treat "find the current build
> instructions in `docs/` or the README" as part of the exercise — reading a real project's
> docs is a skill. Ask Claude Code to help you interpret build errors.

## 3. Run it on recorded data (no car needed)

openpilot logs let you **replay** a real route through the stack:

```bash
# Explore the developer tools
ls tools/
# The replay + UI tools let you play a route and watch the model's outputs.
# Follow tools/replay/README.md for the current invocation.
```

**Do:** replay a route and watch the visualizer. Can you spot where the model's predicted
path is drawn? That drawing code is a great place to start reading.

## 4. Find your way around

Practice landing in the code:

- Where is a message **schema** defined? (look in `cereal/`)
- Where does the model's output get turned into a **plan**? (look in `selfdrive/`)
- Pick one process and trace: what does it subscribe to, and what does it publish?

Ask Claude Code things like: *"In this openpilot checkout, which process publishes the
`modelV2` message and which ones subscribe to it? Show me the files."*

## 5. Scout a good-first-issue (prep for Day 30)

- Read `CONTRIBUTING.md` in the openpilot repo — **follow its rules exactly**; every project
  has its own. Note how they want commits, tests, and PRs formatted.
- Browse Issues filtered by labels like `good first issue` / `bounties`, and recent PRs to see
  what "mergeable" looks like there.
- Good starter contributions are often: a small bug fix with a test, a docs/typo fix, improved
  error messages, or a tooling/dev-experience nit. Avoid touching safety-critical control logic
  as your first PR.

**Do:** pick one candidate issue and write, in your own words: *what's wrong, where the code
likely lives, and how you'd prove your fix works.* That's your Day 30 starting point.

## Check yourself

- Why is openpilot split into many processes instead of one big loop? (Fault isolation,
  scheduling, and the pub-sub logging that makes replay possible — all Week 2 ideas.)
- How does a model trained in PyTorch end up running fast enough on the device? (Week 4.)
- If a process misses its rate deadline, where would you start looking?

## Where this goes next

**Day 30 — Open your first PR:** fork ✓, pick the issue, write the fix *and a test*, run the
project's checks, and submit — following openpilot's `CONTRIBUTING.md` to the letter. You now
have every prerequisite to contribute for real.

---

*This lesson references the independent open-source project
[commaai/openpilot](https://github.com/commaai/openpilot). Commands and layout there change
over time — when in doubt, that repo's own docs are the source of truth.*
