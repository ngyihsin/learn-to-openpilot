"""Day 08 homework — CPU schedulers.

An operating system with one CPU and many ready processes has to decide *who runs next*.
You'll implement three classic scheduling policies and measure how they treat processes
differently. The metrics helper is written for you — your job is to produce the correct
**timeline** of who runs when.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Process:
    pid: str
    arrival: int   # tick at which the process becomes ready
    burst: int     # ticks of CPU time it needs


@dataclass(frozen=True)
class Segment:
    """A contiguous slice of CPU time given to one process: [start, end)."""
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
    """Merge adjacent same-process segments and compute per-process metrics.

    Provided for you — you only have to build the ``segments`` list correctly.
        completion = when the process last ran
        turnaround = completion - arrival
        waiting    = turnaround - burst
    """
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
    """First-Come, First-Served: run processes in arrival order, non-preemptive.

    If the CPU would be idle (the next process hasn't arrived yet), jump time forward.
    """
    # TODO:
    #   - sort a copy of processes by (arrival, pid)
    #   - walk a `time` cursor; for each process, start = max(time, arrival),
    #     append Segment(pid, start, start + burst), advance time
    #   - return finalize(processes, segments)
    raise NotImplementedError


def sjf(processes: list[Process]) -> ScheduleResult:
    """Shortest-Job-First, non-preemptive: among processes that have already arrived,
    always run the one with the smallest burst next (tie-break: earlier arrival, then pid)."""
    # TODO:
    #   - repeatedly pick, from the not-yet-run processes with arrival <= time,
    #     the one with the smallest (burst, arrival, pid)
    #   - if none have arrived yet, jump time to the next arrival
    #   - run it to completion, append its Segment, advance time
    raise NotImplementedError


def round_robin(processes: list[Process], quantum: int) -> ScheduleResult:
    """Round-Robin: each process runs for at most ``quantum`` ticks, then goes to the
    back of the ready queue. Processes that arrive during a quantum join the queue
    *before* the process that was just preempted is re-added."""
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    # TODO:
    #   - keep remaining[pid] = burst, a deque `ready`, and an index into arrivals
    #   - enqueue everything that has arrived by `time`; if the queue is empty, jump
    #     time to the next arrival
    #   - pop a process, run min(quantum, remaining), append a Segment, advance time
    #   - enqueue any processes that arrived by the new time, then re-add the current
    #     process if it still has time left
    raise NotImplementedError
