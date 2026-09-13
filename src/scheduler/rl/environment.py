"""Gymnasium environment for RL-based CPU Scheduling."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scheduler.models import Process
from scheduler.workload import clone_workload
from scheduler.simulation.engine import GanttEvent, SimulationResult


class CPUScheduleEnv(gym.Env):
    """
    RL Environment for CPU Scheduling.
    
    Observation Space: Box(low=0, high=inf, shape=(max_processes, 5), dtype=np.float32)
        Features per process: [is_present, is_ready, remaining_time, priority, current_wait_time]
        
    Action Space: Discrete(max_processes)
        Action represents the index of the process to schedule for the next tick.
    """
    
    metadata = {"render_modes": ["console"]}

    def __init__(self, max_processes: int = 10):
        super().__init__()
        
        self.max_processes = max_processes
        self.action_space = spaces.Discrete(self.max_processes)
        self.observation_space = spaces.Box(
            low=0.0, high=1e5, 
            shape=(self.max_processes, 5), 
            dtype=np.float32
        )
        
        # Internal state
        self.workload: List[Process] = []
        self.current_time = 0
        self.completed_count = 0
        self.gantt_chart: List[GanttEvent] = []
        self.current_event: Optional[GanttEvent] = None
        self.last_action: Optional[int] = None

    def set_workload(self, processes: List[Process]) -> None:
        """Inject a specific workload into the environment before reset."""
        if len(processes) > self.max_processes:
            raise ValueError(f"Workload size ({len(processes)}) exceeds max_processes ({self.max_processes})")
        self.workload = clone_workload(processes)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        
        if not hasattr(self, 'workload') or not self.workload:
            self.workload = []
            
        for p in self.workload:
            p.reset()
            
        self.current_time = 0
        self.completed_count = 0
        self.gantt_chart = []
        self.current_event = None
        self.last_action = None
        
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one CPU tick (time step)."""
        reward = 0.0
        is_valid = False
        target_process = None
        
        # 1. Determine if action is valid
        if action < len(self.workload):
            target_process = self.workload[action]
            if target_process.arrival_time <= self.current_time and target_process.remaining_time > 0:
                is_valid = True

        # 2. Execute process or idle
        if is_valid and target_process is not None:
            # Handle Gantt chart tracking
            if self.last_action != action:
                if self.current_event is not None:
                    self.current_event.end_time = self.current_time
                    self.gantt_chart.append(self.current_event)
                
                self.current_event = GanttEvent(target_process.pid, self.current_time, self.current_time)
                self.last_action = action

            # Initialize start metrics if first time running
            if target_process.start_time is None:
                target_process.start_time = self.current_time
                target_process.response_time = self.current_time - target_process.arrival_time

            # Tick
            target_process.remaining_time -= 1
            
            # Completion check
            if target_process.remaining_time == 0:
                target_process.completion_time = self.current_time + 1
                target_process.turnaround_time = target_process.completion_time - target_process.arrival_time
                target_process.waiting_time = target_process.turnaround_time - target_process.burst_time
                self.completed_count += 1
                
                if self.current_event is not None:
                    self.current_event.end_time = self.current_time + 1
                    self.gantt_chart.append(self.current_event)
                    self.current_event = None
                    self.last_action = None
        else:
            # Invalid action chosen: CPU idles and agent is penalized heavily
            reward -= 10.0
            if self.current_event is not None:
                self.current_event.end_time = self.current_time
                self.gantt_chart.append(self.current_event)
                self.current_event = None
                self.last_action = None

        # 3. Calculate waiting penalties
        ready_queue_size = 0
        for p in self.workload:
            if p.arrival_time <= self.current_time and p.remaining_time > 0:
                if p != target_process or not is_valid:
                    ready_queue_size += 1
                    
        reward -= ready_queue_size
        self.current_time += 1
        
        # 4. Check termination
        terminated = self.completed_count == len(self.workload) and len(self.workload) > 0
        truncated = False 
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Construct the 2D observation matrix."""
        obs = np.zeros((self.max_processes, 5), dtype=np.float32)
        for i, p in enumerate(self.workload):
            is_present = 1.0
            is_ready = 1.0 if (p.arrival_time <= self.current_time and p.remaining_time > 0) else 0.0
            
            wait_time = 0.0
            if p.arrival_time <= self.current_time and p.remaining_time > 0:
                active_execution = (p.burst_time - p.remaining_time)
                wait_time = (self.current_time - p.arrival_time) - active_execution

            obs[i] = [
                is_present, is_ready,
                float(p.remaining_time), float(p.priority), float(wait_time)
            ]
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return diagnostic info, computing final metrics if terminated."""
        info: Dict[str, Any] = {"current_time": self.current_time}
        
        if self.completed_count == len(self.workload) and len(self.workload) > 0:
            # Calculate context switches exactly like BaseScheduler does
            switches = 0
            for i in range(1, len(self.gantt_chart)):
                if self.gantt_chart[i].pid != self.gantt_chart[i - 1].pid:
                    switches += 1
                    
            info["simulation_result"] = SimulationResult(
                processes=self.workload,
                gantt_chart=self.gantt_chart,
                context_switches=switches
            )
        return info