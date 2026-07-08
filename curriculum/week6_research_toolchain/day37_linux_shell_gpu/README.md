# Day 37 — Linux, the Shell & Your GPU Environment

> **Week 6 · Research Toolchain** — every ML paper's code runs on Linux, in a terminal, usually on a
> GPU. Today you get fluent with the ground you'll stand on for the rest of your research.
>
> *A guided, hands-on lesson: you **run** the commands and read the output. No `homework.py`, no
> auto-grader — the terminal is the exercise.*

## Why today matters

When you clone a paper's repo in Week 7 (YOLO, SAM, Grounding DINO), the instructions will assume you
can already: move around the shell, make an isolated Python environment, install dependencies, run a
script, and tell whether a GPU is available. None of that is hard — but if it's unfamiliar, every
paper becomes a wall. Today removes that wall. This is your advisor's Stage 1, tool #1: **Linux/Ubuntu
basics** and the environment your CUDA work will live in.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **Shell / terminal** | The text program where you type commands (`bash`, `zsh`). |
| **Path** | Where you are (`pwd`) and where files live (`/home/you/project`). `.` = here, `..` = up one. |
| **Virtual environment (venv)** | A private Python sandbox per project, so one project's packages don't break another's. |
| **pip** | Python's package installer: `pip install numpy`. |
| **GPU / CUDA** | The graphics card that makes deep learning fast, and NVIDIA's software to use it. |

## Do this — run each block and read the output

**1 · Where am I, and what's here?**
```bash
pwd            # print working directory — your current location
ls -la         # list files here, including hidden ones (names starting with .)
cd ..          # go up one directory ; then cd back with:  cd -
```

**2 · What Python do I have?** (Research needs Python 3.10+.)
```bash
python3 --version
which python3   # the full path to the python that runs
```

**3 · Make an isolated environment** (do this *per project* — never install into system Python):
```bash
python3 -m venv .venv          # create a sandbox named .venv
source .venv/bin/activate      # turn it on — your prompt now shows (.venv)
python -m pip install --upgrade pip
pip install numpy              # installs INTO the sandbox only
python -c "import numpy; print('numpy from', numpy.__file__)"
deactivate                     # turn the sandbox off when done
```
> Why bother? Two papers often need *different* versions of the same library. A venv per project keeps
> them from fighting. It's the #1 habit that saves reproducibility headaches.

**4 · Do I have a GPU, and can PyTorch see it?** (Fine if the answer is "no" — you'll start on CPU.)
```bash
nvidia-smi                                   # NVIDIA driver + GPU status. "command not found" = no NVIDIA GPU.
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
```
> `torch.cuda.is_available()` is the line you'll put at the top of nearly every script:
> `device = 'cuda' if torch.cuda.is_available() else 'cpu'`. Code that respects it runs on both a
> laptop and a GPU server unchanged.

**5 · Run a script and watch a long job.**
```bash
echo "print('hello from a script')" > hello.py
python hello.py
# For jobs that take minutes/hours, keep them alive if your connection drops:
#   tmux new -s train    # start a named session ; detach with Ctrl-b then d ; return with: tmux attach -t train
```

## Check yourself

- Why install packages into a **venv** instead of the system Python? What breaks if you don't?
- A script hard-codes `device = 'cuda'`. What happens when a labmate runs it on a CPU-only laptop? How
  does `torch.cuda.is_available()` fix it?
- `nvidia-smi` says "command not found." Does that mean your code won't run — or just that it'll run on
  CPU (slower)?

## Where this shows up later

Day 38 puts your project under **Git**; Day 39 wraps the whole environment in **Docker** so it runs
identically anywhere; Day 40 uses all of it to reproduce a real paper. In Week 7, `device = 'cuda' if
torch.cuda.is_available() else 'cpu'` is the first line of every model you run.

**Next:** Day 38 — Git & version control for research.
