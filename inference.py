"""Command Line Interface for the CPU Scheduling RL Platform."""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from scheduler.models import Process
from scheduler.workload import generate_workload, load_workload_from_json, clone_workload
from scheduler.algorithms import FCFS, SJF, SRTF, PriorityScheduler, RoundRobin
from scheduler.rl.agent import RLAgent
from scheduler.metrics import calculate_metrics, MetricsReport
from scheduler.visualization import print_process_table, print_metrics_table, plot_gantt_chart, export_to_csv
from scheduler.llm.groq_agent import SchedulingAnalyst


def get_workload(args) -> List[Process]:
    if args.workload:
        print(f"[*] Loading workload from {args.workload}...")
        return load_workload_from_json(args.workload)
    else:
        print(f"[*] Generating random workload (processes={args.processes}, seed={args.seed})...")
        return generate_workload(num_processes=args.processes, seed=args.seed)


def run_single_algorithm(args, workload: List[Process]):
    algorithm_name = args.algorithm.upper()
    print(f"\n--- Running {algorithm_name} Scheduler ---")
    
    if args.algorithm == 'fcfs':
        scheduler = FCFS()
    elif args.algorithm == 'sjf':
        scheduler = SJF()
    elif args.algorithm == 'srtf':
        scheduler = SRTF()
    elif args.algorithm == 'priority':
        scheduler = PriorityScheduler()
    elif args.algorithm == 'rr':
        scheduler = RoundRobin(quantum=args.quantum)
        algorithm_name = f"Round Robin (Q={args.quantum})"
    elif args.algorithm == 'rl':
        print(f"[*] Loading RL model from {args.model}...")
        # Assume the user trained a ceiling model with 10 slots as discussed
        agent = RLAgent(max_processes=max(10, len(workload)), model_path=args.model, seed=args.seed)
        result = agent.predict(workload, deterministic=True)
        metrics = calculate_metrics(result)
        handle_outputs(args, result, metrics, algorithm_name)
        return
    else:
        print(f"[!] Unknown algorithm: {args.algorithm}")
        sys.exit(1)

    result = scheduler.simulate(workload)
    metrics = calculate_metrics(result)
    handle_outputs(args, result, metrics, algorithm_name)


def run_benchmark(args, workload: List[Process]):
    print("\n--- Running Full Benchmark ---")
    
    algorithms = {
        "FCFS": FCFS(),
        "SJF": SJF(),
        "SRTF": SRTF(),
        "Priority": PriorityScheduler(),
        f"RR (Q={args.quantum})": RoundRobin(quantum=args.quantum)
    }

    results_dict: Dict[str, MetricsReport] = {}

    for name, scheduler in algorithms.items():
        current_workload = clone_workload(workload)
        result = scheduler.simulate(current_workload)
        results_dict[name] = calculate_metrics(result)

    print(f"[*] Evaluating RL Agent using {args.model}...")
    try:
        agent = RLAgent(max_processes=max(10, len(workload)), model_path=args.model, seed=args.seed)
        rl_result = agent.predict(clone_workload(workload), deterministic=True)
        rl_metrics = calculate_metrics(rl_result)
        results_dict["RL Agent (PPO)"] = rl_metrics
    except Exception as e:
        print(f"[!] Could not run RL Agent: {e}. Did you train it first?")

    print("\n--- Benchmark Results ---")
    headers = ["Algorithm", "Avg Wait", "Avg Turnaround", "Makespan", "Context Switches"]
    table_data = []
    
    from tabulate import tabulate
    for name, metrics in results_dict.items():
        table_data.append([
            name, 
            f"{metrics.avg_waiting_time:.2f}", 
            f"{metrics.avg_turnaround_time:.2f}", 
            metrics.makespan, 
            metrics.context_switches
        ])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    if args.analyze:
        print("\n[*] Requesting LLM Analysis from Groq...")
        analyst = SchedulingAnalyst()
        analysis = analyst.analyze_benchmark(workload, results_dict)
        print("\n--- Groq LLM Analysis ---")
        print(analysis)
        print("-------------------------\n")


def handle_outputs(args, result, metrics, algo_name):
    print_process_table(result.processes)
    print_metrics_table(metrics, algo_name)
    
    if args.csv:
        export_to_csv(result, metrics, args.csv)
        
    if args.gantt:
        plot_gantt_chart(result, title=f"Gantt Chart: {algo_name}", save_path=args.gantt)
    elif not args.no_display:
        plot_gantt_chart(result, title=f"Gantt Chart: {algo_name}")


def main():
    parser = argparse.ArgumentParser(description="CPU Scheduling Simulator Inference CLI.")
    
    parser.add_argument("--algorithm", type=str, required=True, 
                        choices=["fcfs", "sjf", "srtf", "priority", "rr", "rl", "benchmark"])
    
    parser.add_argument("--workload", type=str, help="Path to JSON workload file")
    parser.add_argument("--processes", type=int, default=5, help="Number of processes for random workload")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quantum", type=int, default=4, help="Time quantum for Round Robin")
    parser.add_argument("--model", type=str, default="models/ppo_scheduler.zip", help="Path to RL model")
    
    parser.add_argument("--gantt", type=str, help="Path to save Gantt chart image")
    parser.add_argument("--csv", type=str, help="Path to save metrics CSV")
    parser.add_argument("--no-display", action="store_true", help="Do not display the Gantt chart GUI window")
    parser.add_argument("--analyze", action="store_true", help="Use Groq LLM to analyze benchmark results")
    
    args = parser.parse_args()
    
    workload = get_workload(args)
    
    if args.algorithm == 'benchmark':
        run_benchmark(args, workload)
    else:
        run_single_algorithm(args, workload)

if __name__ == "__main__":
    main()