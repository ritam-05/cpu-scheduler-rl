"""Script to train the Reinforcement Learning Scheduler."""

import sys
import os
import argparse
from pathlib import Path

# Tell Python to look inside the 'src' folder for our custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from scheduler.rl.agent import RLAgent

def main():
    parser = argparse.ArgumentParser(description="Train the RL CPU Scheduler (PPO).")
    parser.add_argument("--processes", type=int, default=5, help="Max processes per workload (default: 5)")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps (default: 50000)")
    parser.add_argument("--output", type=str, default="models/ppo_scheduler", help="Output path for the model zip file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Initialize Agent
    agent = RLAgent(max_processes=args.processes, seed=args.seed)
    
    # Train
    agent.train(total_timesteps=args.timesteps, save_path=args.output)

if __name__ == "__main__":
    main()