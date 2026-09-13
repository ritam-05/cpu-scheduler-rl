"""Workload generation, validation, and serialization utilities."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scheduler.models import Process


def clone_workload(processes: List[Process]) -> List[Process]:
    """Return a deep copy of processes with reset runtime states.
    
    Prevents side-effects across benchmark algorithm runs.
    """
    clones = copy.deepcopy(processes)
    for p in clones:
        p.reset()
    return clones


def validate_workload(processes: List[Process]) -> None:
    """Ensure workload processes are non-empty and have unique identifiers."""
    if not processes:
        raise ValueError("Workload cannot be empty.")

    pids = [p.pid for p in processes]
    if len(pids) != len(set(pids)):
        duplicates = {pid for pid in pids if pids.count(pid) > 1}
        raise ValueError(f"Duplicate process IDs detected in workload: {duplicates}")


def generate_workload(
    num_processes: int,
    arrival_range: Tuple[int, int] = (0, 10),
    burst_range: Tuple[int, int] = (1, 10),
    priority_range: Tuple[int, int] = (0, 5),
    seed: Optional[int] = None,
) -> List[Process]:
    """Generate a reproducible synthetic CPU workload.

    Args:
        num_processes: Total number of processes to create.
        arrival_range: Min and max inclusive bounds for arrival_time.
        burst_range: Min and max inclusive bounds for burst_time (min >= 1).
        priority_range: Min and max inclusive bounds for priority (0 = highest).
        seed: Random seed for deterministic reproducibility.

    Returns:
        List of generated Process objects sorted by arrival_time.
    """
    if num_processes <= 0:
        raise ValueError("num_processes must be greater than 0.")
    if arrival_range[0] < 0 or arrival_range[0] > arrival_range[1]:
        raise ValueError(f"Invalid arrival_range: {arrival_range}")
    if burst_range[0] <= 0 or burst_range[0] > burst_range[1]:
        raise ValueError(f"Invalid burst_range: {burst_range}. Minimum burst must be >= 1.")
    if priority_range[0] < 0 or priority_range[0] > priority_range[1]:
        raise ValueError(f"Invalid priority_range: {priority_range}")

    rng = random.Random(seed)
    processes: List[Process] = []

    for idx in range(1, num_processes + 1):
        arrival = rng.randint(arrival_range[0], arrival_range[1])
        burst = rng.randint(burst_range[0], burst_range[1])
        prio = rng.randint(priority_range[0], priority_range[1])
        processes.append(
            Process(pid=f"P{idx}", arrival_time=arrival, burst_time=burst, priority=prio)
        )

    # Sort primarily by arrival time, secondarily by PID
    processes.sort(key=lambda p: (p.arrival_time, str(p.pid)))
    validate_workload(processes)
    return processes


def save_workload_to_json(processes: List[Process], filepath: str | Path) -> None:
    """Serialize a list of processes to a formatted JSON file."""
    validate_workload(processes)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [p.to_dict() for p in processes]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_workload_from_json(filepath: str | Path) -> List[Process]:
    """Load and validate a list of processes from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Workload file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of process configurations.")

    processes = [Process.from_dict(item) for item in data]
    validate_workload(processes)
    processes.sort(key=lambda p: (p.arrival_time, str(p.pid)))
    return processes


def manual_workload(specs: List[Dict[str, Any]]) -> List[Process]:
    """Helper to convert manual dict configurations to a validated process list."""
    processes = [Process.from_dict(spec) for spec in specs]
    validate_workload(processes)
    processes.sort(key=lambda p: (p.arrival_time, str(p.pid)))
    return processes