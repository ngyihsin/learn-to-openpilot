"""Day 08 reference solution — FCFS, SJF, and Round-Robin schedulers.

Self-contained: the shared types are re-declared here (identical to homework.py) so this
lesson folder works on its own and the whole-suite grader has no module-name clashes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Process:
    pid: str
    arrival: int
    burst: int


@dataclass(frozen=True)
class Segment:
    pid: str
    start: int
    end: int


@dataclass
class ScheduleResult:
    segments: list[Segment]
    completion: dict[str, int] = field(default_factory=dict)
    turnaround: dict[str, int] = field(default_factory=dict)
    waiting: dict[str, int] = field(default_factory=dict)

    @property
    def avg_waiting(self) -> float:
        return sum(self.waiting.values()) / len(self.waiting)

    @property
    def avg_turnaround(self) -> float:
        return sum(self.turnaround.values()) / len(self.turnaround)


def finalize(processes: list[Process], segments: list[Segment]) -> ScheduleResult:
    merged: list[Segment] = []
    for s in segments:
        if merged and merged[-1].pid == s.pid and merged[-1].end == s.start:
            merged[-1] = Segment(s.pid, merged[-1].start, s.end)
        else:
            merged.append(s)
    by_pid = {p.pid: p for p in processes}
    completion = {p.pid: 0 for p in processes}
    for s in merged:
        completion[s.pid] = max(completion[s.pid], s.end)
    turnaround = {pid: completion[pid] - by_pid[pid].arrival for pid in completion}
    waiting = {pid: turnaround[pid] - by_pid[pid].burst for pid in completion}
    return ScheduleResult(merged, completion, turnaround, waiting)


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
