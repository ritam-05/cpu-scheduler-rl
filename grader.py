"""Graders for CPU scheduling tasks. Each returns score in [0, 1]."""

from __future__ import annotations

from statistics import pstdev
from typing import Dict, List


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _avg_wait(log: Dict) -> float:
    completed = log.get("completed", [])
    if not completed:
        return 1e9
    return sum(float(p["waiting_time"]) for p in completed) / len(completed)


def _priority_alignment(log: Dict) -> float:
    steps = log.get("steps", [])
    if not steps:
        return 0.0

    considered = 0
    aligned = 0
    for step in steps:
        state = step.get("state", {})
        queue = state.get("queue", [])
        selected = step.get("info", {}).get("selected_pid")
        if not queue or selected is None:
            continue
        considered += 1
        min_prio = min(int(p["priority"]) for p in queue)
        selected_prio = None
        for p in queue:
            if str(p["pid"]) == str(selected):
                selected_prio = int(p["priority"])
                break
        if selected_prio is not None and selected_prio == min_prio:
            aligned += 1
    if considered == 0:
        return 0.0
    return aligned / considered


def _fairness_from_waiting(log: Dict) -> float:
    waits = [float(p["waiting_time"]) for p in log.get("completed", [])]
    if not waits:
        return 0.0
    if len(waits) == 1:
        return 1.0
    spread = pstdev(waits)
    # Lower spread is fairer.
    return 1.0 / (1.0 + spread)


def grade_task1_short_job(log: Dict) -> float:
    """Reward short average waiting time (SJF-like behavior)."""
    avg_wait = _avg_wait(log)
    score = 1.0 / (1.0 + avg_wait)
    return _clip01(score)


def grade_task2_priority(log: Dict) -> float:
    """Balance priority compliance and anti-starvation fairness."""
    p_score = _priority_alignment(log)
    f_score = _fairness_from_waiting(log)
    return _clip01(0.65 * p_score + 0.35 * f_score)


def grade_task3_mixed(log: Dict) -> float:
    """Blend waiting time, CPU utilization, and throughput."""
    metrics = log.get("metrics", {})
    avg_wait = float(metrics.get("avg_waiting_time", 1e9))
    utilization = float(metrics.get("cpu_utilization", 0.0))
    throughput = float(metrics.get("throughput", 0.0))

    wait_component = 1.0 / (1.0 + avg_wait)
    throughput_component = throughput / (throughput + 0.5)
    score = 0.45 * wait_component + 0.35 * utilization + 0.20 * throughput_component
    return _clip01(score)


def grade_log(task_grader_key: str, log: Dict) -> float:
    """Dispatch to grader by task key."""
    mapping = {
        "task1": grade_task1_short_job,
        "task2": grade_task2_priority,
        "task3": grade_task3_mixed,
    }
    grader_fn = mapping.get(task_grader_key)
    if grader_fn is None:
        return 0.0
    return float(grader_fn(log))
