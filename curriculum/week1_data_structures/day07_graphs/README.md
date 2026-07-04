# Day 07 — Graphs, BFS & DFS

> **Week 1 · Data Structures** — the most general structure, and the two searches that unlock it.

## Why today matters

Roads, dependencies, social networks, state machines, and the grid a robot plans through are all
**graphs**: nodes connected by edges. Two traversals cover an enormous amount of ground:

- **BFS** explores outward in rings, so on an unweighted graph it finds the **shortest path** (fewest edges).
- **DFS** dives deep before backtracking — the natural tool for reachability, cycle detection, and ordering.

Today you implement both, use BFS to find shortest paths in a graph, and then apply the exact
same idea to **pathfinding on a grid with obstacles** — the toy version of what a motion planner does.

## Learning goals

By the end you can:

- Represent a graph as an adjacency list and traverse it with BFS (queue) and DFS (stack/recursion).
- Reconstruct a shortest path by tracking each node's parent during BFS.
- Apply BFS to a 2-D grid to route around walls.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — watch BFS color a graph in
   distance rings, and see a grid with the shortest path drawn around the obstacles.
2. **Homework (~70 min).** Implement `bfs`, `dfs`, `shortest_path`, and `grid_shortest_path`. The
   `Graph` container is provided.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 7`.

## Hints

- BFS = **queue** (`deque`, pop from the left). DFS = **stack** or recursion. Swapping the data
  structure is literally the only difference between the two searches.
- Mark a node visited *when you enqueue it*, not when you dequeue it — otherwise it can get added
  twice.
- For shortest paths, keep a `parent` dict as you discover nodes; when you reach the goal, walk
  parents backward and reverse. Grid cells `(row, col)` are just graph nodes with up/down/left/right edges.
- `neighbors()` is sorted, so your BFS/DFS orders are deterministic — the grader checks exact orders.

## Check yourself

- Why does BFS find a shortest path but DFS generally doesn't?
- What changes if edges have *weights* (some roads longer than others)? (That's Dijkstra —
  a BFS where the queue becomes the priority queue you built on Day 06.)
- On a 1000×1000 grid, why is BFS's O(V+E) totally fine?

## Where this shows up later

This is the capstone of Week 1: it uses the queue (Day 03), could use the heap (Day 06) for
weighted paths, and previews the state-space search ideas behind planning. In openpilot, routing
and planning are graph/grid search at heart.

**Next:** Week 2 — Operating Systems (Day 08, Processes & Scheduling).
