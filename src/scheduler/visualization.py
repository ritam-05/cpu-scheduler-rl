"""Visualization utilities for CLI tables, Gantt charts, and CSV exports."""

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tabulate import tabulate

from scheduler.models import Process
from scheduler.metrics import MetricsReport
from scheduler.simulation.engine import SimulationResult


def print_process_table(processes: List[Process]) -> None:
    """Print a clean CLI table showing per-process execution metrics."""
    headers = [
        "PID", "Arrival", "Burst", "Priority", 
        "Start", "Completion", "Wait", "Turnaround", "Response"
    ]
    
    table_data = []
    for p in processes:
        table_data.append([
            p.pid, p.arrival_time, p.burst_time, p.priority,
            p.start_time, p.completion_time, 
            p.waiting_time, p.turnaround_time, p.response_time
        ])
        
    print("\n--- Per-Process Metrics ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_metrics_table(metrics: MetricsReport, algorithm_name: str = "Algorithm") -> None:
    """Print a clean CLI table showing aggregate system metrics."""
    data = [
        ["Algorithm", algorithm_name],
        ["Avg Waiting Time", f"{metrics.avg_waiting_time:.2f}"],
        ["Avg Turnaround Time", f"{metrics.avg_turnaround_time:.2f}"],
        ["Avg Response Time", f"{metrics.avg_response_time:.2f}"],
        ["Makespan", f"{metrics.makespan}"],
        ["CPU Utilization", f"{metrics.cpu_utilization:.2f}%"],
        ["Throughput", f"{metrics.throughput:.4f} proc/tick"],
        ["Context Switches", f"{metrics.context_switches}"]
    ]
    
    print("\n--- Aggregate System Metrics ---")
    print(tabulate(data, tablefmt="simple"))


def plot_gantt_chart(result: SimulationResult, title: str = "CPU Scheduling Gantt Chart", save_path: str = None) -> None:
    """Generate and optionally save a Matplotlib Gantt chart of the execution timeline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Assign a unique, consistent color to each PID
    unique_pids = sorted(list(set(p.pid for p in result.processes)), key=lambda x: str(x))
    colors = list(mcolors.TABLEAU_COLORS.values())
    pid_colors = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_pids)}
    
    # Track which labels we've added to the legend to avoid duplicates
    legend_added = set()

    # Plot each execution block
    for event in result.gantt_chart:
        # broken_barh takes a list of (start, duration) tuples and a (y-bottom, height) tuple
        ax.broken_barh(
            [(event.start_time, event.duration)], 
            (10, 9), 
            facecolors=pid_colors[event.pid], 
            edgecolor='black',
            label=event.pid if event.pid not in legend_added else ""
        )
        legend_added.add(event.pid)
        
        # Add text label in the middle of the block if it's wide enough
        if event.duration >= 1:
            ax.text(
                event.start_time + event.duration / 2, 14.5, 
                str(event.pid), 
                ha='center', va='center', color='white', fontweight='bold', fontsize=9
            )

    # Formatting the plot
    ax.set_ylim(5, 25)
    ax.set_xlim(0, max((e.end_time for e in result.gantt_chart), default=10) + 1)
    ax.set_xlabel('Time (Ticks)')
    ax.set_yticks([])
    ax.set_title(title)
    
    # Grid and legend
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300)
        print(f"\n[Gantt Chart saved to {save_path}]")
    else:
        plt.show()


def export_to_csv(result: SimulationResult, metrics: MetricsReport, filepath: str) -> None:
    """Export per-process metrics and aggregate metrics to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write Aggregate Metrics
        writer.writerow(["--- Aggregate Metrics ---"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Avg Waiting Time", metrics.avg_waiting_time])
        writer.writerow(["Avg Turnaround Time", metrics.avg_turnaround_time])
        writer.writerow(["Avg Response Time", metrics.avg_response_time])
        writer.writerow(["Makespan", metrics.makespan])
        writer.writerow(["CPU Utilization (%)", metrics.cpu_utilization])
        writer.writerow(["Throughput", metrics.throughput])
        writer.writerow(["Context Switches", metrics.context_switches])
        writer.writerow([])
        
        # Write Per-Process Metrics
        writer.writerow(["--- Per-Process Metrics ---"])
        writer.writerow([
            "PID", "Arrival Time", "Burst Time", "Priority", 
            "Start Time", "Completion Time", 
            "Waiting Time", "Turnaround Time", "Response Time"
        ])
        for p in result.processes:
            writer.writerow([
                p.pid, p.arrival_time, p.burst_time, p.priority,
                p.start_time, p.completion_time,
                p.waiting_time, p.turnaround_time, p.response_time
            ])
            
    print(f"\n[Metrics exported to {filepath}]")