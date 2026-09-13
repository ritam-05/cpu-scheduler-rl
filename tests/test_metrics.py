"""Unit tests for metrics calculations."""

from scheduler.models import Process
from scheduler.simulation.engine import SimulationResult
from scheduler.metrics import calculate_metrics

def test_metrics_calculation():
    """Verify that aggregate metrics are calculated correctly from mocked process data."""
    # Mocking completed processes
    p1 = Process(pid="P1", arrival_time=0, burst_time=5)
    p1.start_time = 0
    p1.completion_time = 5
    p1.response_time = 0
    p1.turnaround_time = 5
    p1.waiting_time = 0

    p2 = Process(pid="P2", arrival_time=1, burst_time=4)
    p2.start_time = 5
    p2.completion_time = 9
    p2.response_time = 4
    p2.turnaround_time = 8
    p2.waiting_time = 4

    # 2 processes, total burst = 9
    # Arrival min = 0, Completion max = 9 -> Makespan = 9
    # CPU Utilization = (9 / 9) * 100 = 100.0%
    # Avg Wait = (0 + 4) / 2 = 2.0
    # Avg Turnaround = (5 + 8) / 2 = 6.5
    # Avg Response = (0 + 4) / 2 = 2.0

    result = SimulationResult(
        processes=[p1, p2],
        gantt_chart=[],  # Not needed for basic metrics calculation
        context_switches=1
    )

    metrics = calculate_metrics(result)

    assert metrics.avg_waiting_time == 2.0
    assert metrics.avg_turnaround_time == 6.5
    assert metrics.avg_response_time == 2.0
    assert metrics.makespan == 9
    assert metrics.cpu_utilization == 100.0
    assert metrics.throughput == (2 / 9)
    assert metrics.context_switches == 1