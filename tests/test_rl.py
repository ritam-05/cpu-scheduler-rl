"""Tests for the Gymnasium RL Environment."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from scheduler.models import Process
from scheduler.rl.environment import CPUScheduleEnv

def test_gym_environment_compliance():
    """Verify that the environment strictly adheres to the Gymnasium API standard."""
    env = CPUScheduleEnv(max_processes=5)
    # Gymnasium's built-in checker throws exceptions if spaces, reset, or step are malformed
    check_env(env, warn=True, skip_render_check=True)


def test_env_logic_and_rewards():
    """Verify state transitions, rewards, and valid/invalid action handling."""
    env = CPUScheduleEnv(max_processes=3)
    processes = [
        Process(pid="P1", arrival_time=0, burst_time=2, priority=0),
        Process(pid="P2", arrival_time=1, burst_time=2, priority=0),
    ]
    env.set_workload(processes)
    
    obs, info = env.reset()
    assert info["current_time"] == 0
    
    # Tick 0: Only P1 is ready. We choose action 0 (P1).
    # P1 is valid. No other processes waiting. Penalty should be 0.
    obs, reward, terminated, _, _ = env.step(0)
    assert reward == 0.0
    assert not terminated
    assert obs[0][2] == 1.0  # P1 remaining time dropped to 1
    
    # Tick 1: P2 has arrived. Both P1 and P2 are ready. 
    # We choose an invalid action (2), which corresponds to an empty slot.
    # We should get a heavy penalty (-10) + penalty for 2 processes waiting (-2). Total = -12.
    obs, reward, terminated, _, info = env.step(2)
    assert reward == -12.0
    assert info["current_time"] == 2
    
    # Tick 2: Finish P1. P2 is waiting. Penalty for P2 waiting = -1.
    obs, reward, terminated, _, _ = env.step(0)
    assert reward == -1.0
    assert obs[0][2] == 0.0  # P1 is finished
    
    # Tick 3 & 4: Finish P2
    obs, reward, terminated, _, _ = env.step(1)
    obs, reward, terminated, _, info = env.step(1)
    
    assert terminated
    assert "simulation_result" in info
    assert info["simulation_result"].context_switches >= 1