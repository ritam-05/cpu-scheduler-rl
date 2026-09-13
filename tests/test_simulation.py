"""Tests for the base simulation engine."""

from typing import List, Optional
from scheduler.models import Process
from scheduler.simulation.engine import BaseScheduler


class DummyFCFSScheduler(BaseScheduler):
    """A minimal First-Come-First-Served scheduler strictly for testing the engine."""
    def __init__(self):
        self.ready_queue: List[Process] = []

    def on_arrivals(self, arriving_processes: List[Process]) -> None:
        self.ready_queue.extend(arriving_processes)

    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        # If currently running a process, keep running it (non-preemptive)
        if current_process is not None:
            return current_process
        # Otherwise, pick the first in queue
        if self.ready_queue:
            return self.ready_queue[0]
        return None

    def on_process_completion(self, process: Process) -> None:
        self.ready_queue.remove(process)


def test_engine_basic_execution():
    """Ensure engine accurately computes wait times and turnaround times."""
    scheduler = DummyFCFSScheduler()
    processes = [
        Process(pid="P1", arrival_time=0, burst_time=3),
        Process(pid="P2", arrival_time=1, burst_time=2),
    ]
    
    result = scheduler.simulate(processes)
    
    p1 = next(p for p in result.processes if p.pid == "P1")
    p2 = next(p for p in result.processes if p.pid == "P2")
    
    # P1 runs 0 to 3
    assert p1.start_time == 0
    assert p1.completion_time == 3
    assert p1.turnaround_time == 3
    assert p1.waiting_time == 0
    
    # P2 runs 3 to 5
    assert p2.start_time == 3
    assert p2.completion_time == 5
    assert p2.turnaround_time == 4  # 5 (completion) - 1 (arrival)
    assert p2.waiting_time == 2     # 3 (start) - 1 (arrival)
    
    assert len(result.gantt_chart) == 2
    assert result.context_switches == 1


def test_engine_idle_cpu_handling():
    """Ensure engine properly handles gaps where no process is ready."""
    scheduler = DummyFCFSScheduler()
    processes = [
        Process(pid="P1", arrival_time=0, burst_time=2),
        Process(pid="P2", arrival_time=5, burst_time=2), # Gap from 2 to 5
    ]
    
    result = scheduler.simulate(processes)
    
    assert len(result.gantt_chart) == 2
    
    # First block: P1 from 0 to 2
    assert result.gantt_chart[0].pid == "P1"
    assert result.gantt_chart[0].start_time == 0
    assert result.gantt_chart[0].end_time == 2
    
    # Second block: P2 from 5 to 7
    assert result.gantt_chart[1].pid == "P2"
    assert result.gantt_chart[1].start_time == 5
    assert result.gantt_chart[1].end_time == 7
    
    # Context switches across an idle gap count as a switch
    assert result.context_switches == 1