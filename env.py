"""OpenEnv-style CPU scheduling environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Process:
    """Mutable process state used by the scheduler simulation."""

    pid: str
    arrival_time: int
    burst_time: int
    priority: int
    remaining_time: int
    completion_time: Optional[int] = None


class CPUSchedulerEnv:
    """Simple non-preemptive, single-core scheduling environment.

    OpenEnv-style API:
    - reset()
    - step(action)
    - get_state()
    """

    def __init__(self, processes: List[Dict], task_name: str = "unknown") -> None:
        self._seed_processes = processes
        self.task_name = task_name
        self.reset()

    def reset(self) -> Dict:
        """Reset environment state and return the initial observable state."""
        self.current_time = 0
        self.cpu_state = "idle"
        self.idle_time = 0
        self.total_waiting_time = 0

        self.pending: List[Process] = sorted(
            [
                Process(
                    pid=str(p["pid"]),
                    arrival_time=int(p["arrival_time"]),
                    burst_time=int(p["burst_time"]),
                    priority=int(p["priority"]),
                    remaining_time=int(p["burst_time"]),
                )
                for p in self._seed_processes
            ],
            key=lambda x: (x.arrival_time, x.pid),
        )
        self.ready_queue: List[Process] = []
        self.completed: List[Process] = []
        self.steps_log: List[Dict] = []

        self._admit_arrivals()
        return self.get_state()

    def _admit_arrivals(self) -> None:
        while self.pending and self.pending[0].arrival_time <= self.current_time:
            self.ready_queue.append(self.pending.pop(0))

    def get_state(self) -> Dict:
        """Return the current state with queue details, time, and CPU status."""
        queue_state = []
        for p in sorted(
            self.ready_queue, key=lambda x: (x.priority, x.remaining_time, x.pid)
        ):
            queue_state.append(
                {
                    "pid": p.pid,
                    "arrival_time": p.arrival_time,
                    "burst_time": p.burst_time,
                    "remaining_time": p.remaining_time,
                    "priority": p.priority,
                    "wait_time": max(0, self.current_time - p.arrival_time),
                }
            )

        return {
            "task": self.task_name,
            "current_time": self.current_time,
            "cpu_state": self.cpu_state,
            "queue": queue_state,
            "pending_count": len(self.pending),
            "completed_count": len(self.completed),
        }

    def _heuristic_sjf(self) -> Optional[str]:
        if not self.ready_queue:
            return None
        best = min(
            self.ready_queue,
            key=lambda p: (p.remaining_time, p.priority, p.arrival_time, p.pid),
        )
        return best.pid

    def _pick_process(self, action: Optional[str]) -> Tuple[Optional[Process], bool]:
        """Return chosen process and whether fallback was needed."""
        if not self.ready_queue:
            return None, False

        if action is None:
            action = self._heuristic_sjf()

        for idx, proc in enumerate(self.ready_queue):
            if proc.pid == str(action):
                return self.ready_queue.pop(idx), False

        fallback_pid = self._heuristic_sjf()
        for idx, proc in enumerate(self.ready_queue):
            if proc.pid == fallback_pid:
                return self.ready_queue.pop(idx), True
        return None, True

    def step(self, action: Optional[str]):
        """Apply scheduling action and advance simulation.

        Reward: -(waiting_time + idle_time_step) + completion_bonus
        """
        self._admit_arrivals()
        pre_state = self.get_state()

        if not self.ready_queue:
            self.cpu_state = "idle"
            self.current_time += 1
            self.idle_time += 1
            reward = -1.0
            self._admit_arrivals()
            done = not self.ready_queue and not self.pending
            info = {
                "event": "idle",
                "selected_pid": None,
                "waiting_time": 0,
                "turnaround_time": 0,
                "fallback_used": False,
            }
            self.steps_log.append(
                {
                    "state": pre_state,
                    "action": None,
                    "reward": reward,
                    "info": info,
                }
            )
            return self.get_state(), reward, done, info

        self.cpu_state = "busy"
        selected, used_fallback = self._pick_process(action)
        if selected is None:
            # Defensive path: no selectable process after queue mutation.
            self.current_time += 1
            self.idle_time += 1
            reward = -1.0
            done = not self.ready_queue and not self.pending
            info = {
                "event": "idle_due_to_invalid_action",
                "selected_pid": None,
                "waiting_time": 0,
                "turnaround_time": 0,
                "fallback_used": True,
            }
            self.steps_log.append(
                {
                    "state": pre_state,
                    "action": action,
                    "reward": reward,
                    "info": info,
                }
            )
            return self.get_state(), reward, done, info

        waiting_time = max(0, self.current_time - selected.arrival_time)
        run_for = selected.remaining_time
        self.current_time += run_for
        selected.remaining_time = 0
        selected.completion_time = self.current_time
        self.completed.append(selected)

        self.total_waiting_time += waiting_time
        completion_bonus = 5.0
        reward = -float(waiting_time) + completion_bonus
        turnaround_time = selected.completion_time - selected.arrival_time

        self._admit_arrivals()
        done = not self.ready_queue and not self.pending

        info = {
            "event": "completed",
            "selected_pid": selected.pid,
            "waiting_time": waiting_time,
            "turnaround_time": turnaround_time,
            "fallback_used": used_fallback,
        }
        self.steps_log.append(
            {
                "state": pre_state,
                "action": selected.pid,
                "reward": reward,
                "info": info,
            }
        )
        return self.get_state(), reward, done, info

    def get_metrics(self) -> Dict:
        total_processes = len(self.completed)
        if total_processes == 0:
            return {
                "avg_waiting_time": 0.0,
                "avg_turnaround_time": 0.0,
                "cpu_utilization": 0.0,
                "throughput": 0.0,
                "makespan": max(1, self.current_time),
            }

        total_turnaround = sum(
            p.completion_time - p.arrival_time
            for p in self.completed
            if p.completion_time is not None
        )
        makespan = max(1, self.current_time)
        busy_time = max(0, makespan - self.idle_time)

        return {
            "avg_waiting_time": self.total_waiting_time / total_processes,
            "avg_turnaround_time": total_turnaround / total_processes,
            "cpu_utilization": busy_time / makespan,
            "throughput": total_processes / makespan,
            "makespan": makespan,
        }

    def get_log(self) -> Dict:
        """Return full log consumed by graders."""
        completed_serialized = []
        for p in self.completed:
            if p.completion_time is None:
                continue
            turnaround = p.completion_time - p.arrival_time
            waiting = turnaround - p.burst_time
            completed_serialized.append(
                {
                    "pid": p.pid,
                    "arrival_time": p.arrival_time,
                    "burst_time": p.burst_time,
                    "priority": p.priority,
                    "completion_time": p.completion_time,
                    "waiting_time": waiting,
                    "turnaround_time": turnaround,
                }
            )

        return {
            "task": self.task_name,
            "steps": self.steps_log,
            "completed": completed_serialized,
            "metrics": self.get_metrics(),
            "total_processes": len(self._seed_processes),
        }
