# Day 08 — Processes, Threads & CPU Scheduling

> **Week 2 · Operating Systems** — your first look at the OS's hardest, most constant job: deciding who runs next.

## Why today matters

Your laptop runs hundreds of processes on a handful of CPU cores. Only one process can use a
core at a time, so the OS is constantly making a choice: *of everything that's ready to run,
who gets the CPU for the next slice of time?* That choice — the **scheduling policy** — decides
whether your machine feels snappy or sluggish, and whether a real-time system meets its deadlines.

openpilot is a soft real-time system: several processes (camera, model, planner, controls) must
each run at a fixed rate. Understanding scheduling is how you reason about "why did this loop
miss its deadline?"

## Learning goals

By the end you can:

- Explain the difference between a **process** and a **thread**, and what "ready", "running",
  and "waiting" states mean.
- Trace three scheduling policies by hand: **FCFS**, **SJF**, and **Round-Robin**.
- Compute **waiting time** and **turnaround time**, and explain the trade-offs each policy makes
  (throughput vs. fairness vs. responsiveness).

## Do this

1. **Concept + visualization (~25 min).** Run the notebook:
   ```bash
   jupyter lab lesson.ipynb
   ```
   You'll render **Gantt charts** for all three policies on the same workload and *see* why
   Round-Robin feels responsive while SJF finishes short jobs fastest.

2. **Homework (~75 min).** Implement `fcfs`, `sjf`, and `round_robin` in `homework.py`. The
   metrics helper (`finalize`) is written for you — you just build the correct timeline.

3. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 8
   ```

## The workload you'll reason about

| Process | Arrival | Burst |
|---------|--------:|------:|
| P1 | 0 | 5 |
| P2 | 1 | 3 |
| P3 | 2 | 1 |

Work out each timeline on paper first — it makes the code obvious:

- **FCFS** runs them in arrival order. Simple, but the short P3 waits behind the long P1
  (the "convoy effect").
- **SJF** lets the 1-tick P3 jump ahead once it's arrived — lowest average wait, but needs to
  know burst lengths and can starve long jobs.
- **Round-Robin (quantum=2)** slices time so everyone makes progress — responsive, at the cost
  of more context switches.

## Check yourself

- Why does SJF give the lowest average waiting time — and why can't a real OS use it directly?
- What happens to Round-Robin as the quantum → ∞? As it → 1? (One of these is FCFS; the other
  is "maximum fairness, maximum overhead.")
- A **context switch** isn't free. Where would you add that cost to make this simulation more
  realistic?

## Where this shows up later

Day 10 (virtual memory) and Day 11 (threads & the GIL) build on today's process model. In
Week 3 you'll see the actual syscalls (`fork`, `exec`) behind "creating a process," and in
openpilot you'll meet a set of processes scheduled at fixed rates — exactly the timing
reasoning you practiced today.

**Next:** Day 09 — Context switching & the process lifecycle.
