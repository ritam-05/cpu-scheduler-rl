"""Unit tests for classical scheduling algorithms."""

import pytest
from scheduler.models import Process
from scheduler.algorithms import FCFS, SJF, SRTF, PriorityScheduler, RoundRobin

def get_workload():
    """Helper to return a fresh standardized workload for each test."""
    return [
        Process(pid="P1", arrival_time=0, burst_time=8, priority=3),
        Process(pid="P2", arrival_time=1, burst_time=4, priority=1),
        Process(pid="P3", arrival_time=2, burst_time=9, priority=4),
        Process(pid="P4", arrival_time=3, burst_time=5, priority=2),
    ]

def test_fcfs_scheduling():
    scheduler = FCFS()
    result = scheduler.simulate(get_workload())
    
    # FCFS processes strictly in arrival order
    order = [event.pid for event in result.gantt_chart]
    assert order == ["P1", "P2", "P3", "P4"]
    assert result.context_switches == 3

def test_sjf_non_preemptive():
    scheduler = SJF()
    result = scheduler.simulate(get_workload())
    
    # P1 starts at 0. It must finish since SJF is non-preemptive.
    # At t=8 (when P1 finishes), P2, P3, P4 are in queue.
    # Burst times: P2(4), P4(5), P3(9). Order should be P1 -> P2 -> P4 -> P3.
    order = [event.pid for event in result.gantt_chart]
    assert order == ["P1", "P2", "P4", "P3"]

def test_srtf_preemptive():
    scheduler = SRTF()
    result = scheduler.simulate(get_workload())
    
    # P1 starts at t=0. Remaining=8.
    # P2 arrives at t=1. Remaining=4. P1 remaining=7. P2 preempts P1!
    order = [event.pid for event in result.gantt_chart]
    
    # Check that preemption occurred
    assert order[0] == "P1"
    assert order[1] == "P2"
    assert result.context_switches > 3

def test_priority_preemptive():
    scheduler = PriorityScheduler()
    result = scheduler.simulate(get_workload())
    
    # P1 (prio 3) starts at 0.
    # P2 (prio 1) arrives at 1. P2 preempts P1! (Lower = higher prio)
    order = [event.pid for event in result.gantt_chart]
    assert order[0] == "P1"
    assert order[1] == "P2"

def test_round_robin():
    scheduler = RoundRobin(quantum=4)
    result = scheduler.simulate(get_workload())
    
    # P1 has burst 8. It should run for 4 ticks, get preempted, and eventually run again.
    p1_events = [e for e in result.gantt_chart if e.pid == "P1"]
    
    # P1 should have been split into multiple blocks because quantum=4
    assert len(p1_events) > 1
    # The first block must be exactly 4 ticks long
    assert p1_events[0].duration == 4

def test_rr_invalid_quantum():
    with pytest.raises(ValueError):
        RoundRobin(quantum=0)