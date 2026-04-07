"""Task definitions for CPU scheduler evaluation."""

from __future__ import annotations

import random
from typing import Dict, List


def _task_short_job() -> Dict:
    return {
        "name": "short_job_scheduling",
        "description": "Emphasize shortest-job-first behavior to minimize waiting time.",
        "grader": "task1",
        "processes": [
            {"pid": "P1", "arrival_time": 0, "burst_time": 8, "priority": 3},
            {"pid": "P2", "arrival_time": 0, "burst_time": 2, "priority": 2},
            {"pid": "P3", "arrival_time": 1, "burst_time": 1, "priority": 2},
            {"pid": "P4", "arrival_time": 2, "burst_time": 3, "priority": 1},
            {"pid": "P5", "arrival_time": 3, "burst_time": 2, "priority": 3},
        ],
    }


def _task_priority() -> Dict:
    return {
        "name": "priority_scheduling",
        "description": "Respect priority while still reducing starvation risk.",
        "grader": "task2",
        "processes": [
            {"pid": "P1", "arrival_time": 0, "burst_time": 7, "priority": 3},
            {"pid": "P2", "arrival_time": 1, "burst_time": 4, "priority": 1},
            {"pid": "P3", "arrival_time": 2, "burst_time": 5, "priority": 2},
            {"pid": "P4", "arrival_time": 3, "burst_time": 2, "priority": 1},
            {"pid": "P5", "arrival_time": 4, "burst_time": 1, "priority": 3},
            {"pid": "P6", "arrival_time": 6, "burst_time": 3, "priority": 2},
        ],
    }


def _task_mixed(seed: int = 42) -> Dict:
    rng = random.Random(seed)
    processes: List[Dict] = []
    for idx in range(1, 11):
        processes.append(
            {
                "pid": f"P{idx}",
                "arrival_time": rng.randint(0, 10),
                "burst_time": rng.randint(1, 9),
                "priority": rng.randint(1, 4),
            }
        )
    processes.sort(key=lambda p: (p["arrival_time"], p["pid"]))

    return {
        "name": "mixed_workload",
        "description": "Random arrivals with mixed burst lengths and priorities.",
        "grader": "task3",
        "processes": processes,
    }


def load_tasks() -> List[Dict]:
    """Return all required scheduling tasks."""
    return [_task_short_job(), _task_priority(), _task_mixed()]
