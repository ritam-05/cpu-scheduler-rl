"""Tests for process models and workload generation."""

import tempfile
from pathlib import Path
import pytest

from scheduler.models import Process
from scheduler.workload import (
    clone_workload,
    generate_workload,
    load_workload_from_json,
    manual_workload,
    save_workload_to_json,
    validate_workload,
)


def test_process_validation_success():
    """Verify standard process instantiates properly."""
    p = Process(pid="P1", arrival_time=0, burst_time=5, priority=2)
    assert p.pid == "P1"
    assert p.remaining_time == 5
    assert p.is_completed is False


def test_process_invalid_bounds():
    """Check that illegal process attributes raise ValueError."""
    with pytest.raises(ValueError, match="arrival_time must be >= 0"):
        Process(pid="P_ERR", arrival_time=-1, burst_time=5)

    with pytest.raises(ValueError, match="burst_time must be > 0"):
        Process(pid="P_ERR", arrival_time=0, burst_time=0)

    with pytest.raises(ValueError, match="priority must be >= 0"):
        Process(pid="P_ERR", arrival_time=0, burst_time=5, priority=-2)


def test_workload_seed_reproducibility():
    """Identical seeds must generate exact identical process lists."""
    w1 = generate_workload(num_processes=5, seed=42)
    w2 = generate_workload(num_processes=5, seed=42)
    w3 = generate_workload(num_processes=5, seed=99)

    assert len(w1) == len(w2) == 5
    for p1, p2 in zip(w1, w2):
        assert p1.pid == p2.pid
        assert p1.arrival_time == p2.arrival_time
        assert p1.burst_time == p2.burst_time
        assert p1.priority == p2.priority

    # Different seeds should produce distinct profiles
    different = any(p1.burst_time != p3.burst_time for p1, p3 in zip(w1, w3))
    assert different


def test_json_round_trip():
    """Workloads saved to JSON must preserve all attributes upon load."""
    original = generate_workload(num_processes=6, seed=123)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "test_workload.json"
        save_workload_to_json(original, file_path)
        loaded = load_workload_from_json(file_path)

        assert len(original) == len(loaded)
        for p_orig, p_load in zip(original, loaded):
            assert p_orig.pid == p_load.pid
            assert p_orig.arrival_time == p_load.arrival_time
            assert p_orig.burst_time == p_load.burst_time
            assert p_orig.priority == p_load.priority


def test_workload_cloning_isolation():
    """Ensure mutating cloned processes does not corrupt the source workload."""
    base = generate_workload(num_processes=3, seed=7)
    clones = clone_workload(base)

    clones[0].remaining_time = 0
    clones[0].start_time = 10
    clones[0].completion_time = 20

    assert base[0].remaining_time == base[0].burst_time
    assert base[0].start_time is None
    assert base[0].completion_time is None


def test_duplicate_pid_validation():
    """Ensure workloads with colliding process IDs are rejected."""
    dup_specs = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 4},
        {"pid": "P1", "arrival_time": 2, "burst_time": 3},
    ]
    with pytest.raises(ValueError, match="Duplicate process IDs detected"):
        manual_workload(dup_specs)