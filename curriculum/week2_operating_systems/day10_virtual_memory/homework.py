"""Day 10 homework — Virtual memory: address translation & page replacement.

Every process thinks it has the whole address space to itself. The OS keeps that illusion by
splitting memory into fixed-size **pages** and maintaining a **page table** that maps each
virtual page to a physical frame. When a program touches a page that isn't in physical memory,
that's a **page fault**, and the OS must load it — possibly **evicting** another page to make
room. *Which* page to evict is the page-replacement policy.

You'll implement address translation and two classic policies, FIFO and LRU, then (in the
notebook) watch FIFO fall victim to **Belady's anomaly**: more memory, *more* faults.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class PageFault(Exception):
    """Raised by ``translate`` when a virtual page has no mapping in the page table."""


@dataclass
class ReplacementResult:
    faults: int = 0
    hits: int = 0
    fault_flags: list[bool] = field(default_factory=list)   # was reference i a fault?
    frames_over_time: list[list[int]] = field(default_factory=list)  # frame contents after each ref

    @property
    def fault_rate(self) -> float:
        total = self.faults + self.hits
        return self.faults / total if total else 0.0


def translate(vaddr: int, page_size: int, page_table: dict[int, int]) -> int:
    """Translate a virtual address to a physical address.

        virtual page number (vpn) = vaddr // page_size
        offset                    = vaddr %  page_size
        physical address          = frame * page_size + offset

    Raise ``PageFault`` if the vpn isn't mapped.
    """
    # TODO: split vaddr into vpn and offset, look up the frame, combine, or raise PageFault
    raise NotImplementedError


def fifo(references: list[int], num_frames: int) -> ReplacementResult:
    """FIFO page replacement: when memory is full, evict the page that was loaded *earliest*,
    regardless of how recently it was used.

    ``references`` is the sequence of page numbers the program touches.
    Record, for each reference, whether it faulted and a snapshot of the frames afterward.
    """
    # TODO:
    #   - keep the set/list of pages currently in frames, in load order
    #   - for each page: if present -> hit; else -> fault, and if full evict the oldest
    #   - append a bool to fault_flags and a *copy* of the current frames to frames_over_time
    raise NotImplementedError


def lru(references: list[int], num_frames: int) -> ReplacementResult:
    """LRU page replacement: when memory is full, evict the **least recently used** page.

    A page counts as 'used' on every reference to it (hit or the fault that loads it).
    """
    # TODO: same shape as fifo, but on a hit you must mark that page as most-recently-used,
    #       and evict the page whose last use is oldest
    raise NotImplementedError
