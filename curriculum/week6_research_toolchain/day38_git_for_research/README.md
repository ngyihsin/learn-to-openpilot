# Day 38 — Git & Version Control for Research

> **Week 6 · Research Toolchain** — Git is how you save your work, undo mistakes, collaborate, and
> read *other people's* code history. Every paper repo you touch lives in Git.
>
> *Guided, hands-on: run the commands in a throwaway folder and watch what happens.*

## Why today matters

Research code changes constantly: you try an idea, it fails, you revert; you branch to test a variant
without breaking what works; you read a repo's history to understand *why* a line exists. Git makes all
of that safe. And a research-specific skill most tutorials skip: **never commit datasets or model
weights** — they're huge and belong outside Git. Getting `.gitignore` right is half the job.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **repository (repo)** | A folder Git is tracking, plus its whole history. |
| **commit** | A saved snapshot with a message — a point you can always return to. |
| **branch** | A parallel line of work; try things without touching `main`. |
| **`.gitignore`** | A list of files Git should ignore — data, weights, `.venv`, caches. |
| **remote / clone / push / pull** | The copy on GitHub; download it, send commits up, get commits down. |

## Do this — run each block

**1 · Start a repo and make your first commit.**
```bash
mkdir /tmp/git-practice && cd /tmp/git-practice
git init
echo "# My experiment" > README.md
git status                 # shows README.md as untracked
git add README.md
git commit -m "First commit"
git log --oneline          # your commit, with its short hash
```

**2 · See what changed, then save it.**
```bash
echo "results: 91% accuracy" >> README.md
git diff                   # exactly what changed since the last commit
git add -A && git commit -m "Log first result"
git log --oneline          # two commits now
```

**3 · Branch to try an idea safely.**
```bash
git switch -c try-bigger-model   # new branch, you're now on it
echo "model: resnet50" > config.txt
git add -A && git commit -m "Try resnet50"
git switch main                  # back to main — config.txt is gone here
git log --oneline --all --graph  # see both lines of work
```

**4 · Ignore the things you must NOT commit** (this is the research-critical bit):
```bash
printf "%s\n" ".venv/" "*.pth" "*.ckpt" "data/" "__pycache__/" "*.log" > .gitignore
mkdir data && dd if=/dev/zero of=data/huge.bin bs=1M count=5 2>/dev/null  # a fake 5 MB "dataset"
git status                 # data/ and the .bin do NOT show up — good.
git add -A && git commit -m "Add .gitignore"
```
> Model weights (`.pth`, `.ckpt`) and datasets are often gigabytes. Committing them bloats the repo
> forever (history keeps them even after deletion). Ignore them; share them via a download link instead.

**5 · Read someone else's history** (what you'll do with every paper repo):
```bash
git log --oneline -5       # the last 5 commits and their messages
git blame README.md        # who wrote each line, and in which commit — great for "why is this here?"
```

## Check yourself

- Why is committing a 2 GB weights file a mistake that's hard to fully undo? What do you do instead?
- You're on `main` and want to test a risky change. What's the safe move before you start editing?
- `git diff` vs `git log`: which shows *what changed right now*, and which shows *the history of saved
  snapshots*?

## Where this shows up later

Day 40 has you clone a paper's repo and read its history to reproduce it. In your own research, a clean
Git habit — small commits, good messages, a strict `.gitignore` — is what lets you (and your advisor)
trust and rerun your experiments months later.

**Next:** Day 39 — Docker & reproducible environments.
