"""Tests for the Groq LLM Agent integration."""

import os
from scheduler.models import Process
from scheduler.metrics import MetricsReport
from scheduler.llm.groq_agent import SchedulingAnalyst

def test_mock_mode_fallback(monkeypatch):
    """Ensure the agent correctly falls back to mock mode if no API key is provided."""
    # Force the environment variable to be empty
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    
    analyst = SchedulingAnalyst()
    assert analyst.mock_mode is True
    
    # Create dummy data
    workload = [Process(pid="P1", arrival_time=0, burst_time=5)]
    results = {
        "FCFS": MetricsReport(avg_waiting_time=10.0, avg_turnaround_time=15.0, avg_response_time=10.0, 
                              makespan=15, cpu_utilization=100.0, throughput=0.1, context_switches=1),
        "SJF": MetricsReport(avg_waiting_time=2.0, avg_turnaround_time=7.0, avg_response_time=2.0, 
                             makespan=15, cpu_utilization=100.0, throughput=0.1, context_switches=1)
    }
    
    response = analyst.analyze_benchmark(workload, results)
    
    assert "[MOCK MODE ANALYST REPORT]" in response
    assert "SJF" in response  # It should correctly identify SJF as having the lowest wait time