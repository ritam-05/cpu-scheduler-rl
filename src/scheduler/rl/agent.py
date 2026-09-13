"""RL Agent wrapper and training environment."""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from stable_baselines3 import PPO

from scheduler.models import Process
from scheduler.workload import generate_workload
from scheduler.rl.environment import CPUScheduleEnv
from scheduler.simulation.engine import SimulationResult


class RandomizedCPUEnv(CPUScheduleEnv):
    """An environment that generates a new random workload on every reset.
    
    This prevents the RL agent from overfitting to a single workload during training.
    """
    def __init__(self, max_processes: int = 5, seed: Optional[int] = None):
        super().__init__(max_processes=max_processes)
        self.env_seed = seed
        self.current_episode = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        run_seed = seed if seed is not None else (
            self.env_seed + self.current_episode if self.env_seed is not None else None
        )
        
        new_workload = generate_workload(
            num_processes=self.max_processes,
            arrival_range=(0, 10),
            burst_range=(1, 10),
            priority_range=(0, 4),
            seed=run_seed
        )
        self.set_workload(new_workload)
        self.current_episode += 1
        
        return super().reset(seed=seed, options=options)


class RLAgent:
    """Wrapper for the PPO RL Agent."""
    
    def __init__(self, max_processes: int = 5, model_path: Optional[str] = None, seed: int = 42):
        self.max_processes = max_processes
        self.seed = seed
        self.env = RandomizedCPUEnv(max_processes=max_processes, seed=seed)
        
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=self.env)
            print(f"[RLAgent] Loaded existing model from {model_path}")
        else:
            self.model = PPO(
                "MlpPolicy", 
                self.env, 
                verbose=1,
                seed=self.seed,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                ent_coef=0.01
            )
            print("[RLAgent] Initialized fresh PPO model.")

    def train(self, total_timesteps: int, save_path: str) -> None:
        """Train the agent in the randomized environment."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Starting PPO Training for {total_timesteps} timesteps ---")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
        
        self.model.save(save_path)
        print(f"--- Training Complete. Model saved to {save_path}.zip ---\n")

    def predict(self, workload: List[Process], deterministic: bool = True) -> SimulationResult:
        """Use the trained model to schedule a specific workload."""
        eval_env = CPUScheduleEnv(max_processes=self.max_processes)
        eval_env.set_workload(workload)
        
        obs, info = eval_env.reset()
        terminated = False
        
        while not terminated:
            action, _states = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))

        return info["simulation_result"]