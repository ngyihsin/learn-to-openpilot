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
| 02 | Linked lists (singly, doubly) | notebook + pytest | A doubly-linked list with O(1) insert/delete | ⬜ |
| 03 | Stacks & queues | pytest | Stack, queue, and a ring buffer | ⬜ |
| 04 | Hash maps | notebook + pytest | An open-addressing hash table | ⬜ |
| 05 | Trees & BSTs | notebook + pytest | A binary search tree with traversals | ⬜ |
| 06 | Heaps & priority queues | notebook + pytest | A binary heap (you'll reuse it in Day 8!) | ⬜ |
| 07 | Graphs, BFS & DFS | notebook + pytest | Graph + shortest-path on a grid | ⬜ |

## Week 2 — Operating Systems
*Goal: understand what happens between your code and the hardware.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 08 | **Processes, threads & CPU scheduling** | notebook + pytest | FCFS / SJF / Round-Robin schedulers + metrics | ✅ |
| 09 | Context switching & the process lifecycle | notebook | A state-machine simulator | ⬜ |
| 10 | Virtual memory & paging | notebook + pytest | A page-table + FIFO/LRU page-replacement sim | ⬜ |
| 11 | Threads & the GIL | notebook + pytest | Threaded vs. multiprocess benchmark | ⬜ |
| 12 | Synchronization: locks, semaphores | notebook + pytest | A bounded-buffer producer/consumer | ⬜ |
| 13 | Deadlock & race conditions | notebook | Reproduce and fix a data race | ⬜ |
| 14 | IPC, pipes & the file system | notebook + pytest | A tiny key-value store over a pipe | ⬜ |

## Week 3 — Systems Programming
*Goal: touch the metal — memory, syscalls, and the C/Python boundary openpilot lives on.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 15 | Memory model: stack, heap, pointers | notebook + pytest | A bump allocator (in Python, then read C) | ⬜ |
| 16 | C for Python programmers | pytest | Port a hot function to C | ⬜ |
| 17 | Calling C from Python (ctypes / cffi) | pytest | Wrap a C function, benchmark the speedup | ⬜ |
| 18 | Syscalls & the kernel boundary | notebook | Trace a program with `strace` | ⬜ |
| 19 | Sockets & networking | pytest | A minimal TCP echo server/client | ⬜ |
| 20 | Profiling & performance | notebook + pytest | Find and fix a 100× slowdown | ⬜ |
| 21 | Build systems (Make, scons, CMake) | notebook | Build a multi-file C project (openpilot uses scons) | ⬜ |

## Week 4 — PyTorch / ExecuTorch
*Goal: train a model, then ship it to run on-device the way openpilot does.*

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 22 | **Tensors & autograd** | notebook + pytest | Gradient descent from scratch, then with autograd | ✅ |
| 23 | `nn.Module` & the training loop | notebook + pytest | Train a classifier on a toy dataset | ⬜ |
| 24 | CNNs & computer vision | notebook + pytest | A small CNN on grayscale road-sign images | ⬜ |
| 25 | Datasets, DataLoaders & augmentation | pytest | An efficient input pipeline | ⬜ |
| 26 | Exporting models (TorchScript / ONNX / `torch.export`) | notebook + pytest | Export & re-load a trained model | ⬜ |
| 27 | **ExecuTorch**: on-device inference | notebook + pytest | Lower a model to ExecuTorch and run it | ⬜ |
| 28 | Quantization & making models fast | notebook + pytest | Quantize a model, measure size/latency | ⬜ |

## Days 29–30 — The openpilot On-Ramp
*Goal: go from "I understand the pieces" to "I opened a PR."*

| Day | Topic | Format | You'll do | Status |
|-----|-------|--------|-----------|--------|
| 29 | **openpilot architecture & building it** | guided README | Tour the stack, build it, run a replay | ✅ |
| 30 | Find a good-first-issue & open a PR | guided README | Pick an issue, write a test, submit a PR | ⬜ |

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
