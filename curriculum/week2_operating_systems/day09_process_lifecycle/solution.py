"""Day 09 reference solution — process lifecycle state machine."""
from __future__ import annotations


class InvalidTransition(Exception):
    """Raised when an event isn't legal from the current state."""


TRANSITIONS: dict[tuple[str, str], str] = {
    ("new", "admit"): "ready",
    ("ready", "dispatch"): "running",
    ("running", "timeout"): "ready",
    ("running", "wait"): "waiting",
    ("waiting", "wakeup"): "ready",
    ("running", "exit"): "terminated",
}


class ProcessStateMachine:
    def __init__(self, pid: str) -> None:
        self.pid = pid
        self.state = "new"
        self.history: list[str] = ["new"]
        self.context_switches = 0

    def can(self, event: str) -> bool:
        return (self.state, event) in TRANSITIONS

    def on(self, event: str) -> str:
        if not self.can(event):
            raise InvalidTransition(
                f"process {self.pid}: cannot '{event}' from state '{self.state}'"
            )
        self.state = TRANSITIONS[(self.state, event)]
        self.history.append(self.state)
        if event == "dispatch":
            self.context_switches += 1
        return self.state

    def run(self, events: list[str]) -> str:
        for event in events:
            self.on(event)
        return self.state

    def is_terminated(self) -> bool:
        return self.state == "terminated"
