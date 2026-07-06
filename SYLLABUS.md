# The 30-Day Syllabus

Each day ≈ 2–3 hours: a concept lesson you can *see*, then a graded homework you *run*.
Days marked ✅ are fully built. The rest are scaffolded slots being filled in from
[`templates/lesson_template/`](templates/lesson_template).

The arc is deliberate: the data structures you build in Week 1 are the ones the OS uses in
Week 2; the OS concepts (memory, scheduling, syscalls) are what systems programming in Week 3
manipulates directly; and PyTorch/ExecuTorch in Week 4 sits on top of *all* of it. By the
time you reach openpilot, nothing in its stack is a black box.

---

## Week 1 — Data Structures & Algorithms
*Goal: build the core containers from scratch and reason about cost with Big-O.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 01 | **Dynamic arrays & amortized analysis** | notebook + pytest | A resizable array with amortized O(1) append | ✅ |
| 02 | **Linked lists (singly, doubly)** | notebook + pytest | A doubly-linked list with O(1) insert/delete | ✅ |
| 03 | **Stacks & queues** | pytest | Stack, queue, and a ring buffer | ✅ |
| 04 | **Hash maps** | notebook + pytest | An open-addressing hash table | ✅ |
| 05 | **Trees & BSTs** | notebook + pytest | A binary search tree with traversals | ✅ |
| 06 | **Heaps & priority queues** | notebook + pytest | A binary heap (you'll reuse it in Day 8!) | ✅ |
| 07 | **Graphs, BFS & DFS** | notebook + pytest | Graph + shortest-path on a grid | ✅ |

## Week 2 — Operating Systems
*Goal: understand what happens between your code and the hardware.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 08 | **Processes, threads & CPU scheduling** | notebook + pytest | FCFS / SJF / Round-Robin schedulers + metrics | ✅ |
| 09 | **Context switching & the process lifecycle** | notebook + pytest | A state-machine simulator | ✅ |
| 10 | **Virtual memory & paging** | notebook + pytest | A page-table + FIFO/LRU page-replacement sim | ✅ |
| 11 | **Threads & the GIL** | notebook + pytest | Threaded vs. multiprocess benchmark | ✅ |
| 12 | **Synchronization: locks, semaphores** | notebook + pytest | A bounded-buffer producer/consumer | ✅ |
| 13 | **Deadlock & race conditions** | notebook + pytest | Deadlock detector + reproduce/fix a race | ✅ |
| 14 | **IPC, pipes & the file system** | notebook + pytest | A tiny key-value store over a pipe | ✅ |

## Week 3 — Systems Programming
*Goal: touch the metal — memory, syscalls, and the C/Python boundary openpilot lives on.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 15 | **Memory model: stack, heap, pointers** | notebook + pytest | A bump allocator | ✅ |
| 16 | **C for Python programmers** | pytest | Port hot functions to C (compiled + called via ctypes) | ✅ |
| 17 | **Calling C from Python (ctypes / cffi)** | pytest | Wrap C array functions, marshal data across | ✅ |
| 18 | **Syscalls & the kernel boundary** | notebook + pytest | Raw file I/O via os.open/read/write/lseek | ✅ |
| 19 | **Sockets & networking** | pytest | A minimal framed TCP echo server/client | ✅ |
| 20 | **Profiling & performance** | notebook + pytest | Find and fix an O(n²) slowdown | ✅ |
| 21 | **Build systems (Make, scons, CMake)** | notebook + pytest | Dependency graph + incremental rebuild logic | ✅ |

## Week 4 — PyTorch / ExecuTorch
*Goal: train a model, then ship it to run on-device the way openpilot does.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 22 | **Tensors & autograd** | notebook + pytest | Gradient descent from scratch, then with autograd | ✅ |
| 23 | **`nn.Module` & the training loop** | notebook + pytest | Train an MLP classifier on a toy dataset | ✅ |
| 24 | **CNNs & computer vision** | notebook + pytest | A small CNN on synthetic images | ✅ |
| 25 | **Datasets, DataLoaders & augmentation** | pytest | A custom Dataset + DataLoader pipeline | ✅ |
| 26 | **Exporting models (TorchScript / ONNX / `torch.export`)** | notebook + pytest | Export & re-load a trained model | ✅ |
| 27 | **ExecuTorch**: on-device inference | notebook + pytest | Lower a model with torch.export and run it | ✅ |
| 28 | **Quantization & making models fast** | notebook + pytest | Quantize a model, measure size + error | ✅ |

## Days 29–30 — The openpilot On-Ramp
*Goal: go from "I understand the pieces" to "I opened a PR."*

| Day | Topic | Format | You'll do | Status |
|-----|-------|--------|-----------|--------|
| 29 | **openpilot architecture & building it** | guided README | Tour the stack, build it, run a replay | ✅ |
| 30 | **Find a good-first-issue & open a PR** | guided README | Pick an issue, write a test, submit a PR | ✅ |

---

## Extension — Beyond Day 30: toward computer-vision / VLM research

*These weeks extend the original 30 to serve a specific goal: taking a learner from ML foundations to
being able to reproduce papers and run modern vision / vision-language models (YOLO, SAM, Grounding
DINO) on driving datasets (KITTI, NuScenes) — the on-ramp to a master's research project.*

### Week 5 — ML & Deep Learning Foundations
*Goal: the concepts the advanced courses assume — what "learning" is, and the two mechanisms
(deep nets, attention) everything modern is built from.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 31 | **numpy & array thinking** | notebook + pytest | Vectorized math, masks, axes, argmin, seeded rng | ✅ |
| 32 | **The ML framing: model, loss, generalization** | notebook + pytest | Train/val split, polynomial fit, model selection | ✅ |
| 33 | **Regression & gradient descent** | notebook + pytest | A linear model trained by hand-written gradient descent | ✅ |
| 34 | **Classification: softmax & cross-entropy** | notebook + pytest | Softmax, cross-entropy, one-hot, argmax, accuracy | ✅ |
| 35 | **Neural networks & backpropagation** | notebook + pytest | A 2-layer net with hand-derived backprop (gradient-checked) | ✅ |
| 36 | **Self-attention & Transformers** | notebook + pytest | Scaled dot-product self-attention from scratch | ✅ |

### Week 6 — Research Toolchain
*Goal: run other people's code, on real hardware, reproducibly. (Advisor Stage 1.)*

| Day | Topic | Format | You'll be able to | Status |
|-----|-------|--------|-------------------|--------|
| 37 | **Linux, the shell & your GPU environment** | guided | Navigate the shell, make a venv, check CPU/GPU/CUDA | ✅ |
| 38 | **Git & version control for research** | guided | Branch, commit, `.gitignore` big files, read a repo's history | ✅ |
| 39 | **Docker & reproducible environments** | guided | Run a project in a container that works anywhere | ✅ |
| 40 | **Reproducing a paper end-to-end** | guided | Clone → env → weights → inference → match the reported result | ✅ |
| 41 | **Structuring a PyTorch research project** | pytest | Configs, seeds, logging, checkpoints — reproducibility utils | ✅ |

### Week 7 — Computer Vision & Vision-Language Models
*Goal: run the open-source models the lab uses — YOLO, SAM, Grounding DINO — on driving data. (Advisor Stage 2.)*

| Day | Topic | Format | You'll be able to | Status |
|-----|-------|--------|-------------------|--------|
| 42 | **Detection fundamentals: boxes, IoU & NMS** | pytest | Compute IoU, run non-max suppression, convert box formats | ✅ |
| 43 | **YOLO: running a real detector** | guided | Clone YOLO, run inference, read its boxes/scores/labels | ✅ |
| 44 | **Segmentation with SAM 2 / SAM 3** | guided | Prompt SAM with a point/box, get masks, understand the output | ✅ |
| 45 | **Open-vocabulary detection: Grounding DINO** | guided | Detect objects from a *text prompt*, not a fixed class list | ✅ |
| 46 | **CLIP & vision-language models** | pytest | Match images to text with embeddings + cosine similarity | ✅ |
| 47 | **Datasets: KITTI & NuScenes** | guided | Understand driving-dataset formats; load labels for inference | ✅ |

### Week 8 — Paper Reading & the Research On-ramp
*Goal: read top-venue papers, find a direction, evaluate honestly. (Advisor Stage 3.)*

| Day | Topic | Format | You'll be able to | Status |
|-----|-------|--------|-------------------|--------|
| 48 | **How to read a paper (CVPR/ICCV/ECCV)** | guided | Read a paper in 3 passes; extract the contribution & the catch | ✅ |
| 49 | **Finding a topic & the literature map** | guided | Turn a vague interest into a question; map the related work | ✅ |
| 50 | **Evaluating models: metrics done right** | pytest | Compute precision/recall/F1; avoid the accuracy trap | ✅ |
| 51 | **Experiments, tracking & ablations** | guided | Design a fair comparison; change one thing at a time; log it | ✅ |
| 52 | **Capstone: your mini research project** | guided | Scope a small reproduce-then-tweak project end-to-end | ✅ |

---

## How the pieces connect to openpilot

| What you learn here | Where it shows up in openpilot |
|---------------------|--------------------------------|
| Ring buffers, heaps (W1) | Message queues & the pub/sub logging system |
| Scheduling & real-time (W2) | Multiple processes at fixed rates on the device |
| Shared memory & IPC (W2–3) | `cereal`/`msgq` passing sensor data between processes |
| C ↔ Python, scons (W3) | The C++ core with Python controls & tooling |
| PyTorch → ExecuTorch, quantization (W4) | The driving model that runs on the comma device |

## Progress tracking

`tools/grade.py status` reads which homeworks pass and prints your completion map, so you
always know where you are in the 30 days.
