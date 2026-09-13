"""End-to-End Dynamic CPU Scheduling Experiment Pipeline."""

import sys
import os
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from scheduler.workload import generate_workload, clone_workload
from scheduler.algorithms import FCFS, SJF, SRTF, PriorityScheduler, RoundRobin
from scheduler.rl.agent import RLAgent
from scheduler.metrics import calculate_metrics
from scheduler.llm.groq_agent import SchedulingAnalyst
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(description="Run a Dynamic End-to-End Scheduling Experiment.")
    parser.add_argument("--timesteps", type=int, default=250000, help="Timesteps for on-the-fly RL training (default: 25000)")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific random seed")
    parser.add_argument("--no-analyze", action="store_true", help="Skip the Groq LLM analysis")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(1, 100000)
    rng = random.Random(seed)
    num_processes = rng.randint(10, 30)
    
    print(f"\n[Step 1] Generating fresh workload with {num_processes} processes (Seed: {seed})...")
    workload = generate_workload(
        num_processes=num_processes,
        arrival_range=(0, 30),
        burst_range=(1, 15),
        priority_range=(0, 5),
        seed=seed
    )

    print(f"\n[Step 2] Training RL Agent for {num_processes} processes ({args.timesteps} timesteps)...")
    model_path = f"models/dynamic_ppo_{num_processes}"
    agent = RLAgent(max_processes=num_processes, seed=seed)
    agent.train(total_timesteps=args.timesteps, save_path=model_path)

    print("\n[Step 3] Running Classical Algorithms vs Fresh RL Agent...")
    algorithms = {
        "FCFS": FCFS(),
        "SJF": SJF(),
        "SRTF": SRTF(),
        "Priority": PriorityScheduler(),
        "RR (Q=4)": RoundRobin(quantum=4)
    }

    results_dict = {}
    
    for name, scheduler in algorithms.items():
        res = scheduler.simulate(clone_workload(workload))
        results_dict[name] = calculate_metrics(res)

    rl_res = agent.predict(clone_workload(workload), deterministic=True)
    results_dict["RL Agent (PPO)"] = calculate_metrics(rl_res)

    print("\n--- Final Benchmark Results ---")
    headers = ["Algorithm", "Avg Wait", "Avg Turnaround", "Makespan", "Context Switches"]
    table_data = []
    
    sorted_results = sorted(results_dict.items(), key=lambda item: item[1].avg_waiting_time)
    
    for name, metrics in sorted_results:
        table_data.append([
            name, 
            f"{metrics.avg_waiting_time:.2f}", 
            f"{metrics.avg_turnaround_time:.2f}", 
            metrics.makespan, 
            metrics.context_switches
        ])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    if not args.no_analyze:
        print("\n[Step 4] Requesting LLM Analysis from Groq...")
        analyst = SchedulingAnalyst()
        analysis = analyst.analyze_benchmark(workload, results_dict)
        print("\n--- Groq LLM Analysis ---")
        print(analysis)
        print("-------------------------\n")

if __name__ == "__main__":
    main()