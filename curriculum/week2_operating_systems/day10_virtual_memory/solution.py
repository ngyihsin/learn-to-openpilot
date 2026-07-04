"""Day 10 reference solution — address translation, FIFO & LRU page replacement.

Self-contained: the shared types are re-declared here (identical to homework.py) so this
lesson folder works on its own and the whole-suite grader has no module-name clashes.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field


class PageFault(Exception):
    """Raised by ``translate`` when a virtual page has no mapping in the page table."""


@dataclass
class ReplacementResult:
    faults: int = 0
    hits: int = 0
    fault_flags: list[bool] = field(default_factory=list)
    frames_over_time: list[list[int]] = field(default_factory=list)

    @property
    def fault_rate(self) -> float:
        total = self.faults + self.hits
        return self.faults / total if total else 0.0


def translate(vaddr: int, page_size: int, page_table: dict[int, int]) -> int:
    vpn, offset = divmod(vaddr, page_size)
    if vpn not in page_table:
        raise PageFault(f"virtual page {vpn} is not mapped")
    return page_table[vpn] * page_size + offset


def fifo(references: list[int], num_frames: int) -> ReplacementResult:
    res = ReplacementResult()
    frames: set[int] = set()
    order: deque[int] = deque()  # load order, oldest at the left
    for page in references:
        if page in frames:
            res.hits += 1
            res.fault_flags.append(False)
        else:
            res.faults += 1
            res.fault_flags.append(True)
            if len(frames) >= num_frames:
                evict = order.popleft()
                frames.discard(evict)
            frames.add(page)
            order.append(page)
        res.frames_over_time.append(list(order))
    return res


def lru(references: list[int], num_frames: int) -> ReplacementResult:
    res = ReplacementResult()
    # OrderedDict as a recency list: left = least recently used, right = most recent.
    frames: "OrderedDict[int, None]" = OrderedDict()
    for page in references:
        if page in frames:
            res.hits += 1
            res.fault_flags.append(False)
            frames.move_to_end(page)          # mark most-recently-used
        else:
            res.faults += 1
            res.fault_flags.append(True)
            if len(frames) >= num_frames:
                frames.popitem(last=False)    # evict least-recently-used
            frames[page] = None
        res.frames_over_time.append(list(frames))
    return res
