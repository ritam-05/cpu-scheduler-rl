"""Run LLM-driven CPU scheduling inference on all tasks.

Required stdout format per episode:
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from agent import LLMPolicyAgent
from env import CPUSchedulerEnv
from grader import grade_log
from tasks import load_tasks


BENCHMARK = os.getenv("BENCHMARK", "cpu_scheduler")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task: Dict, agent: LLMPolicyAgent) -> None:
    env = CPUSchedulerEnv(task["processes"], task_name=task["name"])
    state = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task["name"], env_name=BENCHMARK, model=MODEL_NAME)
    try:
        done = False
        while not done:
            action, _reasoning, fallback_used = agent.select_action(state)
            env_action = None if action == "IDLE" else action

            state, reward, done, info = env.step(env_action)

            steps_taken += 1
            rewards.append(float(reward))

            error = None
            if fallback_used:
                error = "fallback_used"
            if info.get("event") == "idle_due_to_invalid_action":
                error = "invalid_action"

            log_step(
                step=steps_taken,
                action=action,
                reward=float(reward),
                done=done,
                error=error,
            )

        log = env.get_log()
        score = float(grade_log(task["grader"], log))
        score = max(0.0, min(1.0, score))
        success = score > 0.0

    except Exception as exc:
        log_step(
            step=steps_taken + 1, action="ERROR", reward=0.0, done=True, error=str(exc)
        )

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    agent = LLMPolicyAgent(max_retries=3, timeout_seconds=30)
    for task in load_tasks():
        run_task(task, agent)


if __name__ == "__main__":
    main()
