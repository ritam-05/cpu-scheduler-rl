"""Core discrete-time CPU scheduling simulation engine."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from scheduler.models import Process
from scheduler.workload import clone_workload


@dataclass
class GanttEvent:
    """Represents a continuous block of CPU execution for a single process."""
    pid: str | int
    start_time: int
    end_time: int

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time


@dataclass
class SimulationResult:
    """The final result and collected metrics of a scheduling simulation."""
    processes: List[Process]
    gantt_chart: List[GanttEvent]
    context_switches: int


class BaseScheduler(ABC):
    """Abstract base class for all CPU scheduling algorithms.
    
    This engine uses a discrete tick-by-tick simulation. It handles time progression,
    metric tracking, and Gantt chart generation. Subclasses only need to implement 
    the queueing and selection logic.
    """

    def simulate(self, processes: List[Process]) -> SimulationResult:
        """Run the scheduling simulation on the provided workload."""
        # Clone workload to prevent mutating the user's base list
        workload = clone_workload(processes)
        
        # Sort pending by arrival time to easily check arrivals
        pending = sorted(workload, key=lambda p: (p.arrival_time, str(p.pid)))
        
        current_time = 0
        completed_count = 0
        total_processes = len(workload)
        
        gantt_chart: List[GanttEvent] = []
        current_process: Optional[Process] = None
        current_event: Optional[GanttEvent] = None
        
        # Loop until all processes have fully executed
        while completed_count < total_processes:
            # 1. Handle new arrivals at the current tick
            arriving = [p for p in pending if p.arrival_time == current_time]
            for p in arriving:
                pending.remove(p)
            if arriving:
                self.on_arrivals(arriving)
            
            # 2. Ask subclass which process should run right now
            next_process = self.schedule(current_time, current_process)
            
            # 3. Handle state transitions & Gantt tracking
            if next_process != current_process:
                # Close the previous execution block
                if current_event is not None:
                    current_event.end_time = current_time
                    gantt_chart.append(current_event)
                    current_event = None
                
                # Start a new execution block if not idle
                if next_process is not None:
                    current_event = GanttEvent(
                        pid=next_process.pid, 
                        start_time=current_time, 
                        end_time=current_time
                    )
                    
                current_process = next_process

            # 4. Execute one CPU tick for the scheduled process
            if current_process is not None:
                # Record initial start time on very first execution
                if current_process.start_time is None:
                    current_process.start_time = current_time
                    current_process.response_time = current_time - current_process.arrival_time
                    
                current_process.remaining_time -= 1
                
                # 5. Handle completion
                if current_process.remaining_time == 0:
                    current_process.completion_time = current_time + 1
                    current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                    current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                    
                    completed_count += 1
                    
                    # Close Gantt event upon completion
                    if current_event is not None:
                        current_event.end_time = current_time + 1
                        gantt_chart.append(current_event)
                        current_event = None
                    
                    self.on_process_completion(current_process)
                    current_process = None
                    
            # Progress time
            current_time += 1
            
        # Post-simulation metric calculations
        context_switches = self._calculate_context_switches(gantt_chart)
        
        # Sort processes back to original PID order before returning
        workload.sort(key=lambda p: str(p.pid))
        
        return SimulationResult(
            processes=workload,
            gantt_chart=gantt_chart,
            context_switches=context_switches
        )

    def _calculate_context_switches(self, gantt_chart: List[GanttEvent]) -> int:
        """Count the number of times execution transitioned to a different process."""
        switches = 0
        for i in range(1, len(gantt_chart)):
            if gantt_chart[i].pid != gantt_chart[i - 1].pid:
                switches += 1
        return switches

    @abstractmethod
    def on_arrivals(self, arriving_processes: List[Process]) -> None:
        """Invoked when processes arrive. Subclasses should add them to their ready queue."""
        pass

    @abstractmethod
    def schedule(self, current_time: int, current_process: Optional[Process]) -> Optional[Process]:
        """Invoked every tick. Return the Process to execute, or None to idle."""
        pass

    def on_process_completion(self, process: Process) -> None:
        """Optional hook invoked when a process finishes execution."""
        pass