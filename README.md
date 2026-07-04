# learn-openpilot

**A 30-day, hands-on computer-science bootcamp that takes you from data structures
all the way to contributing code to [openpilot](https://github.com/commaai/openpilot).**

You learn by *doing*: every day is a short concept lesson you can **see** (visualizations,
animations, Gantt charts, loss curves) plus a **graded** homework you actually run.
The whole thing is designed to be taught interactively with **Claude Code** sitting next
to you as a tutor — ask it to explain a cell, unstick a failing test, or go deeper.

```
Data Structures  →  Operating Systems  →  Systems Programming  →  PyTorch / ExecuTorch  →  openpilot
    Week 1               Week 2                 Week 3                   Week 4            Days 29–30
```

## Who this is for

You can write basic Python (loops, functions, classes) and want to become someone who can
read a real, large C++/Python robotics codebase and land a pull request. No prior CS degree
required — we build the mental models from scratch and *show* you how they work.

## How a day works

Each `dayNN_*` folder contains:

| File | What it is |
|------|-----------|
| `README.md` | The day's goals, ~15 min read, and how to run everything |
| `lesson.ipynb` | The **concept** notebook — explanations + **visualizations** you run cell by cell |
| `homework.py` | Starter code with `TODO`s — the thing **you** implement |
| `test_homework.py` | Auto-grader. When it's green, you've got it. |

Concept/visualization-heavy days ship as **Jupyter notebooks**; coding & systems days are
**`.py` + `pytest`** so you build muscle memory with real tooling. (See the
[design note](#format-notebooks-vs-py--pytest) below.)

## Quickstart

```bash
git clone https://github.com/ngyihsin/learn-openpilot.git
cd learn-openpilot

python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# See where you are and what's next
python tools/grade.py status

# Open the first lesson's concept notebook
jupyter lab curriculum/week1_data_structures/day01_dynamic_arrays/lesson.ipynb
```

Then do the homework and grade it:

```bash
cd curriculum/week1_data_structures/day01_dynamic_arrays
# ...edit homework.py until...
pytest -q            # or: python ../../../tools/grade.py day 1
```

## Learning with Claude Code

This repo is built to be *taught*. Open it in Claude Code and try prompts like:

- *"Explain the resize logic in cell 4 of today's notebook like I'm 12, then quiz me."*
- *"My `test_homework.py` fails on `test_amortized_growth` — walk me through why, but don't write the fix for me."*
- *"Give me one harder bonus exercise for today and a hint."*
- *"I finished Day 8. Show me how today's scheduler idea shows up in a real Linux kernel."*

There's a suggested tutor system-prompt in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## The 30 days at a glance

See **[SYLLABUS.md](SYLLABUS.md)** for the full day-by-day plan. Short version:

- **Week 1 — Data Structures & Algorithms:** dynamic arrays, linked lists, stacks/queues, hash maps, trees, graphs, Big-O.
- **Week 2 — Operating Systems:** processes & threads, CPU scheduling, virtual memory, concurrency, synchronization, IPC, file systems.
- **Week 3 — Systems Programming:** memory & pointers, C ↔ Python, syscalls, sockets, profiling, build systems.
- **Week 4 — PyTorch / ExecuTorch:** tensors & autograd, training loops, CNNs, model export, on-device inference, quantization.
- **Days 29–30 — openpilot on-ramp:** architecture tour, build & run, find a good-first-issue, open your first PR.

## What's built so far

This is an actively-growing curriculum. Fully-built sample lessons that show the pattern
end-to-end (concept notebook + homework + passing auto-grader):

- ✅ **Week 1 — Data Structures: complete (Days 01–07)**
  - Day 01 Dynamic Arrays · Day 02 Linked Lists · Day 03 Stacks/Queues/Ring Buffer ·
    Day 04 Hash Maps · Day 05 Trees & BSTs · Day 06 Heaps · Day 07 Graphs (BFS/DFS + grid pathfinding)
- ✅ **Week 2 — Operating Systems: complete (Days 08–14)**
  - Day 08 CPU Scheduling · Day 09 Process Lifecycle · Day 10 Virtual Memory · Day 11 Threads & the GIL ·
    Day 12 Synchronization · Day 13 Deadlock · Day 14 IPC & Pipes
- ✅ **Week 3 — Systems Programming: complete (Days 15–21)**
  - Day 15 Memory Model (bump allocator) · Day 16 C for Python · Day 17 ctypes FFI · Day 18 Syscalls ·
    Day 19 Sockets · Day 20 Profiling · Day 21 Build Systems
- ✅ **Day 22 — Tensors & Autograd** (`curriculum/week4_pytorch_executorch/day22_tensors_autograd`)
- ✅ **Day 29 — openpilot on-ramp** (`curriculum/capstone_openpilot/day29_openpilot_onramp`)

Every other day has a slot in `SYLLABUS.md` and is being filled in using the
[`templates/lesson_template/`](templates/lesson_template) pattern. Contributions welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Format: notebooks vs `.py` + pytest

We deliberately mix the two:

- **Notebooks** shine when the point is to *see* something evolve — an array doubling its
  capacity, three schedulers racing, a loss curve descending. You run a cell, a picture
  appears, intuition lands.
- **`.py` + pytest** shines for the actual *engineering* — writing a class, getting edge
  cases right, feeling what real test-driven development is like (which is exactly how you'll
  work in openpilot).

So concept days lean notebook; coding/systems days lean pytest. Most days have both.

## Repo layout

```
learn-openpilot/
├── README.md              ← you are here
├── SYLLABUS.md            ← full 30-day plan
├── CONTRIBUTING.md        ← how to author a lesson (+ Claude Code tutor prompt)
├── requirements.txt
├── pytest.ini
├── Makefile               ← make test / make grade / make notebooks
├── tools/
│   ├── grade.py           ← progress tracker + per-day grader
│   ├── nbbuild.py         ← tiny helper to author .ipynb from plain Python
│   └── build_lessons.py   ← regenerates the sample lesson notebooks
├── templates/
│   └── lesson_template/   ← copy this to start a new day
└── curriculum/
    ├── week1_data_structures/
    ├── week2_operating_systems/
    ├── week3_systems_programming/
    ├── week4_pytorch_executorch/
    └── capstone_openpilot/
```

## License

[MIT](LICENSE). Learn freely, remix freely.

> openpilot is a project of comma.ai and is released under the MIT License. This curriculum
> is an independent educational resource and is not affiliated with or endorsed by comma.ai.
