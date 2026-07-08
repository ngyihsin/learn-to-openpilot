# Day 39 — Docker & Reproducible Environments

> **Week 6 · Research Toolchain** — "it works on my machine" is the enemy of reproducible research.
> Docker packages the *entire* environment — OS, libraries, CUDA — so it runs the same everywhere.
>
> *Guided, hands-on: read a real Dockerfile and run a container.*

## Why today matters

A paper's results depend on more than its code: a specific Python, exact library versions, system
packages, a CUDA version. Reproduce it on a slightly different machine and it breaks. **Docker** solves
this by shipping a whole prepared environment as an **image** you can run anywhere identically. Many
paper repos now ship a `Dockerfile` precisely so you can reproduce them without fighting dependencies —
learning to read and run one removes the single most common reproduction failure.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **image** | A frozen, ready-to-run environment (OS + libs + your code). A blueprint. |
| **container** | A running instance of an image — a lightweight, isolated "computer." |
| **Dockerfile** | The recipe that builds an image, step by step. |
| **registry (Docker Hub)** | Where images are shared: `docker pull python:3.11`. |
| **volume** | A folder shared between your machine and the container (so data/results persist). |

## Do this

**1 · Is Docker installed?** (If not, that's fine — read along; install later from docs.docker.com.)
```bash
docker --version                 # "command not found" -> not installed yet
docker run hello-world           # pulls a tiny image and runs it, if Docker is set up
```

**2 · Read a Dockerfile — the recipe.** A typical ML project's `Dockerfile` looks like this:
```dockerfile
FROM python:3.11-slim            # start from a small Python image
WORKDIR /app                     # work inside /app in the container
COPY requirements.txt .          # copy just deps first (caching)
RUN pip install -r requirements.txt
COPY . .                         # copy the rest of the code
CMD ["python", "infer.py"]       # what to run when the container starts
```
> Read it top-to-bottom as "steps to set up a fresh machine." Each line is a layer Docker caches, so
> rebuilds are fast when only your code changed.

**3 · Run a project in a container** (the pattern you'll use for paper repos):
```bash
# Explore an image interactively without installing anything on your host:
docker run -it --rm python:3.11-slim bash
#   inside the container:  python --version ; exit
# Mount your current folder so the container can read data and write results:
docker run --rm -v "$PWD":/app -w /app python:3.11-slim python --version
```

**4 · GPUs in Docker** (for when you have one): images built for CUDA run with
```bash
# docker run --gpus all -it --rm pytorch/pytorch:latest python -c "import torch; print(torch.cuda.is_available())"
```
> The base image (e.g. `pytorch/pytorch`) already contains the right CUDA — you don't install it yourself.
> That's the whole point: the environment is baked in.

## Check yourself

- What's the difference between an **image** and a **container**? (Blueprint vs running instance.)
- A paper "works in their Docker but not in your venv." What kind of problem is Docker hiding for them?
- Why does a `Dockerfile` copy `requirements.txt` and `pip install` *before* copying the rest of the
  code? (Hint: layer caching — which step changes most often?)

## Where this shows up later

Day 40 reproduces a paper; if it ships a Dockerfile, you'll use exactly these commands. In a lab, you'll
run experiments in containers so results are portable across your laptop, the lab server, and a cloud GPU.

**Next:** Day 40 — Reproducing a paper end-to-end.
