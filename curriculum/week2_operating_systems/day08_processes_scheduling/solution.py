"""Day 08 reference solution — FCFS, SJF, and Round-Robin schedulers."""
from __future__ import annotations

from collections import deque

from homework import Process, Segment, ScheduleResult, finalize  # reuse the shared types

# Note: this import works because pytest runs with the lesson dir on the path via the
# loader in test_homework.py. When authored standalone we re-declare below if needed.


def fcfs(processes: list[Process]) -> ScheduleResult:
    order = sorted(processes, key=lambda p: (p.arrival, p.pid))
    time = 0
    segments: list[Segment] = []
    for p in order:
        start = max(time, p.arrival)
        end = start + p.burst
        segments.append(Segment(p.pid, start, end))
        time = end
    return finalize(processes, segments)


def sjf(processes: list[Process]) -> ScheduleResult:
    remaining = list(processes)
    time = 0
    segments: list[Segment] = []
    while remaining:
        ready = [p for p in remaining if p.arrival <= time]
        if not ready:
            time = min(p.arrival for p in remaining)
            continue
        p = min(ready, key=lambda p: (p.burst, p.arrival, p.pid))
        remaining.remove(p)
        start = max(time, p.arrival)
        end = start + p.burst
        segments.append(Segment(p.pid, start, end))
        time = end
    return finalize(processes, segments)


def round_robin(processes: list[Process], quantum: int) -> ScheduleResult:
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    arrivals = sorted(processes, key=lambda p: (p.arrival, p.pid))
    remaining = {p.pid: p.burst for p in processes}
    ready: deque[str] = deque()
    segments: list[Segment] = []
    time = 0
    i = 0  # next un-enqueued arrival

    def enqueue_arrived(upto: int) -> None:
        nonlocal i
        while i < len(arrivals) and arrivals[i].arrival <= upto:
            ready.append(arrivals[i].pid)
            i += 1

    enqueue_arrived(time)
    while ready or i < len(arrivals):
        if not ready:
            time = arrivals[i].arrival
            enqueue_arrived(time)
            continue
        pid = ready.popleft()
        run = min(quantum, remaining[pid])
        start = time
        time += run
        remaining[pid] -= run
        segments.append(Segment(pid, start, time))
        enqueue_arrived(time)          # newcomers board before the preempted process
        if remaining[pid] > 0:
            ready.append(pid)
    return finalize(processes, segments)
