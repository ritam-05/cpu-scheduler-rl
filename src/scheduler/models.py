"""Data models for CPU scheduling simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Process:
    """Represents a computational task to be scheduled on the CPU.
    
    Priority Convention:
        Lower numerical value = Higher priority (0 is highest priority).
        
    Attributes:
        pid: Unique identifier for the process (e.g., 1, 2 or "P1").
        arrival_time: Time tick when the process enters the ready queue (>= 0).
        burst_time: Total CPU execution cycles required (> 0).
        priority: Priority value (>= 0, default 0).
        remaining_time: Cycles left before completion.
        start_time: Time tick when the process first received CPU execution.
        completion_time: Time tick when remaining_time reached 0.
        waiting_time: Total time spent in ready queue waiting for CPU.
        turnaround_time: Total time from arrival to completion (completion - arrival).
        response_time: Time from arrival to first CPU allocation (start - arrival).
    """

    pid: int | str
    arrival_time: int
    burst_time: int
    priority: int = 0

    # Runtime mutable state
    remaining_time: int = field(init=False)
    start_time: Optional[int] = field(default=None, init=False)
    completion_time: Optional[int] = field(default=None, init=False)
    waiting_time: Optional[int] = field(default=None, init=False)
    turnaround_time: Optional[int] = field(default=None, init=False)
    response_time: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate process attributes and initialize execution state."""
        if self.arrival_time < 0:
            raise ValueError(
                f"Process {self.pid}: arrival_time must be >= 0, got {self.arrival_time}"
            )
        if self.burst_time <= 0:
            raise ValueError(
                f"Process {self.pid}: burst_time must be > 0, got {self.burst_time}"
            )
        if self.priority < 0:
            raise ValueError(
                f"Process {self.pid}: priority must be >= 0, got {self.priority}"
            )

        self.remaining_time = self.burst_time

    def reset(self) -> None:
        """Reset mutable execution metrics to initial state for re-simulation."""
        self.remaining_time = self.burst_time
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
        self.turnaround_time = None
        self.response_time = None

    @property
    def is_completed(self) -> bool:
        """Check if process has finished execution."""
        return self.remaining_time == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize static configuration to dictionary."""
        return {
            "pid": self.pid,
            "arrival_time": self.arrival_time,
            "burst_time": self.burst_time,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Process:
        """Instantiate a Process from dictionary configuration."""
        return cls(
            pid=data["pid"],
            arrival_time=int(data["arrival_time"]),
            burst_time=int(data["burst_time"]),
            priority=int(data.get("priority", 0)),
        )