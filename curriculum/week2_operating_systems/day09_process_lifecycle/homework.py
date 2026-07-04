"""Day 09 homework — the process lifecycle as a state machine.

A process isn't just "running" — it moves through a small set of states as the OS admits it,
gives it the CPU, preempts it, blocks it on I/O, and finally reaps it:

    new --admit--> ready --dispatch--> running --exit--> terminated
                     ^                    | |
                     |  wakeup            | | timeout   (preempted -> back to ready)
                   waiting <---wait-------+ +---------------> ready

Every ``dispatch`` is the OS handing this process the CPU — a **context switch**. Modeling this
as a state machine is exactly how kernels reason about it, and it makes illegal transitions
(like running a terminated process) impossible by construction.

Implement the transition logic and run ``pytest -q``.
"""
from __future__ import annotations


class InvalidTransition(Exception):
    """Raised when an event isn't legal from the current state."""


# (current_state, event) -> next_state.  This encodes the diagram above.
TRANSITIONS: dict[tuple[str, str], str] = {
    ("new", "admit"): "ready",
    ("ready", "dispatch"): "running",
    ("running", "timeout"): "ready",     # preempted (quantum expired)
    ("running", "wait"): "waiting",      # blocked on I/O
    ("waiting", "wakeup"): "ready",      # I/O finished
    ("running", "exit"): "terminated",
}


class ProcessStateMachine:
    def __init__(self, pid: str) -> None:
        self.pid = pid
        self.state = "new"
        self.history: list[str] = ["new"]
        self.context_switches = 0    # count of `dispatch` events (times it got the CPU)

    def can(self, event: str) -> bool:
        """Return True if `event` is legal from the current state."""
        # TODO: is (self.state, event) a key in TRANSITIONS?
        raise NotImplementedError

    def on(self, event: str) -> str:
        """Apply `event`. Update state, append to history, count context switches.
        Raise InvalidTransition if the event isn't legal here. Return the new state."""
        # TODO:
        #   - if not self.can(event): raise InvalidTransition(...)
        #   - move to the next state, append it to history
        #   - if event == "dispatch": self.context_switches += 1
        raise NotImplementedError

    def run(self, events: list[str]) -> str:
        """Apply a whole sequence of events in order; return the final state."""
        # TODO: call self.on(e) for each e
        raise NotImplementedError

    def is_terminated(self) -> bool:
        return self.state == "terminated"
