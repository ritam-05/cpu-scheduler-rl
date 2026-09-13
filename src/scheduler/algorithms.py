"""Classical CPU Scheduling Algorithms."""

from typing import List, Optional
from scheduler.models import Process
from scheduler.simulation.engine import BaseScheduler


class FCFS(BaseScheduler):
    """First-Come, First-Served (Non-preemptive FIFO)."""
    def __init__(self):
        self.queue: List[Process] = []

    def on_arrivals(self, processes: List[Process]) -> None:
        self.queue.extend(processes)

    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        if current_process is not None:
            return current_process  # Non-preemptive
        return self.queue[0] if self.queue else None

    def on_process_completion(self, process: Process) -> None:
        self.queue.remove(process)


class SJF(BaseScheduler):
    """Shortest Job First (Non-preemptive)."""
    def __init__(self):
        self.queue: List[Process] = []

    def on_arrivals(self, processes: List[Process]) -> None:
        self.queue.extend(processes)

    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        if current_process is not None:
            return current_process  # Non-preemptive
        if not self.queue:
            return None
        # Tie-breaker: burst_time -> arrival_time -> PID
        self.queue.sort(key=lambda p: (p.burst_time, p.arrival_time, str(p.pid)))
        return self.queue[0]

    def on_process_completion(self, process: Process) -> None:
        self.queue.remove(process)


class SRTF(BaseScheduler):
    """Shortest Remaining Time First (Preemptive SJF)."""
    def __init__(self):
        self.queue: List[Process] = []

    def on_arrivals(self, processes: List[Process]) -> None:
        self.queue.extend(processes)

    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        if not self.queue:
            return None
        # Preemptive: Re-evaluate the shortest remaining time on every tick
        self.queue.sort(key=lambda p: (p.remaining_time, p.arrival_time, str(p.pid)))
        return self.queue[0]

    def on_process_completion(self, process: Process) -> None:
        self.queue.remove(process)


class PriorityScheduler(BaseScheduler):
    """Priority Scheduling (Preemptive, Lower integer = Higher priority)."""
    def __init__(self):
        self.queue: List[Process] = []

    def on_arrivals(self, processes: List[Process]) -> None:
        self.queue.extend(processes)

    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        if not self.queue:
            return None
        # Sort by priority, then arrival time, then PID
        self.queue.sort(key=lambda p: (p.priority, p.arrival_time, str(p.pid)))
        return self.queue[0]

    def on_process_completion(self, process: Process) -> None:
        self.queue.remove(process)


class RoundRobin(BaseScheduler):
    """Round Robin Scheduling (Preemptive time-sliced)."""
    def __init__(self, quantum: int):
        if quantum <= 0:
            raise ValueError("Time quantum must be a positive integer.")
        self.quantum = quantum
        self.queue: List[Process] = []
        self.current_process_time = 0

    def on_arrivals(self, processes: List[Process]) -> None:
        self.queue.extend(processes)

    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        if current_process is not None:
            self.current_process_time += 1
            if self.current_process_time == self.quantum:
                # Quantum expired, requeue to the back
                self.queue.remove(current_process)
                self.queue.append(current_process)
                self.current_process_time = 0
                return self.queue[0]
            else:
                return current_process  # Continue running

        # No process running currently, pick the next in line
        if self.queue:
            self.current_process_time = 0
            return self.queue[0]
            
        return None

    def on_process_completion(self, process: Process) -> None:
        self.queue.remove(process)
        self.current_process_time = 0