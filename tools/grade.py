#!/usr/bin/env python3
"""Progress tracker + per-day auto-grader for learn-openpilot.

Usage
-----
    python tools/grade.py status      # show your 30-day completion map
    python tools/grade.py day 8       # run just Day 8's auto-grader
    python tools/grade.py list        # list discovered lessons

A day counts as "done" when its ``test_homework.py`` passes.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRICULUM = ROOT / "curriculum"

DAY_RE = re.compile(r"day(\d{2})_")


def discover() -> dict[int, Path]:
    """Map day-number -> lesson directory, sorted."""
    days: dict[int, Path] = {}
    for d in sorted(CURRICULUM.glob("*/day*_*")):
        m = DAY_RE.match(d.name)
        if m and d.is_dir():
            days[int(m.group(1))] = d
    return days


def _has_grader(d: Path) -> bool:
    return (d / "test_homework.py").exists()


def _passes(d: Path) -> bool:
    """Run the day's grader quietly; True if pytest exits 0."""
    if not _has_grader(d):
        return False
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(d), "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def cmd_status() -> int:
    days = discover()
    if not days:
        print("No lessons discovered yet under curriculum/.")
        return 0
    print("\n  learn-openpilot — progress\n  " + "-" * 34)
    done = 0
    for n in range(1, 31):
        d = days.get(n)
        if d is None:
            mark, label = "·", "(not built yet)"
        elif not _has_grader(d):
            mark, label = "◒", d.name + "  (no grader — read-only lesson)"
        elif _passes(d):
            mark, label = "✅", d.name
            done += 1
        else:
            mark, label = "⬜", d.name + "  (homework not passing)"
        print(f"  Day {n:02d}  {mark}  {label}")
    gradable = sum(1 for d in days.values() if _has_grader(d))
    print("  " + "-" * 34)
    print(f"  {done}/{gradable} graded days passing "
          f"({len(days)} lessons built of 30)\n")
    return 0


def cmd_day(arg: str) -> int:
    try:
        n = int(arg)
    except ValueError:
        print(f"Not a day number: {arg!r}")
        return 2
    d = discover().get(n)
    if d is None:
        print(f"Day {n} isn't built yet. See SYLLABUS.md.")
        return 1
    if not _has_grader(d):
        print(f"Day {n} ({d.name}) is a read-only lesson — no auto-grader. "
              f"Read its README.md.")
        return 0
    print(f"Grading Day {n}: {d.name}\n")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(d), "-v"], cwd=ROOT
    ).returncode


def cmd_list() -> int:
    for n, d in sorted(discover().items()):
        kind = "graded" if _has_grader(d) else "read-only"
        print(f"  Day {n:02d}  {d.relative_to(ROOT)}  [{kind}]")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    cmd, *rest = argv
    if cmd == "status":
        return cmd_status()
    if cmd == "list":
        return cmd_list()
    if cmd == "day" and rest:
        return cmd_day(rest[0])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
