"""Calculates aggregate system metrics from a simulation result."""

from dataclasses import dataclass
from typing import List

from scheduler.models import Process
from scheduler.simulation.engine import SimulationResult


@dataclass
class MetricsReport:
    """Aggregate metrics for a completed scheduling simulation."""
    avg_waiting_time: float
    avg_turnaround_time: float
    avg_response_time: float
    makespan: int
    cpu_utilization: float  # Percentage (0.0 to 100.0)
    throughput: float       # Processes completed per unit of time
    context_switches: int


def calculate_metrics(result: SimulationResult) -> MetricsReport:
    """Calculate aggregate system performance metrics from a simulation run."""
    processes = result.processes
    num_processes = len(processes)

    if num_processes == 0:
        return MetricsReport(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)

    # Calculate sums
    total_waiting = sum(p.waiting_time for p in processes if p.waiting_time is not None)
    total_turnaround = sum(p.turnaround_time for p in processes if p.turnaround_time is not None)
    total_response = sum(p.response_time for p in processes if p.response_time is not None)
    
    # Time bounds
    min_arrival = min(p.arrival_time for p in processes)
    max_completion = max(p.completion_time for p in processes if p.completion_time is not None)
    makespan = max_completion - min_arrival

    # Hardware metrics
    total_burst_time = sum(p.burst_time for p in processes)
    cpu_utilization = (total_burst_time / makespan * 100.0) if makespan > 0 else 0.0
    throughput = (num_processes / makespan) if makespan > 0 else 0.0

    return MetricsReport(
        avg_waiting_time=total_waiting / num_processes,
        avg_turnaround_time=total_turnaround / num_processes,
        avg_response_time=total_response / num_processes,
        makespan=makespan,
        cpu_utilization=cpu_utilization,
        throughput=throughput,
        context_switches=result.context_switches
    )