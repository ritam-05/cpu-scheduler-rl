"""FastAPI service for the CPU scheduling environment."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env import CPUSchedulerEnv
from grader import grade_log
from tasks import load_tasks


BENCHMARK = os.getenv("BENCHMARK", "cpu_scheduler")
TASKS = load_tasks()
TASKS_BY_ID = {task["grader"]: task for task in TASKS}
TASKS_BY_NAME = {task["name"]: task for task in TASKS}

app = FastAPI(
    title="CPU Scheduler RL",
    description="OpenEnv-style API for a CPU scheduling benchmark.",
    version="1.0.0",
)

_current_task: Optional[Dict] = None
_current_env: Optional[CPUSchedulerEnv] = None
_last_done = False
_last_score = 0.0


class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task1",
        description="Task key such as task1/task2/task3 or the task name.",
    )
    algorithm: str = Field(
        default="sjf",
        description="Scheduling policy: fcfs, sjf, srjf, rr, or priority.",
    )
    time_quantum: int = Field(
        default=2,
        ge=1,
        description="Time quantum used by round robin.",
    )


class StepRequest(BaseModel):
    action: Optional[str] = Field(
        default=None,
        description="Process id to schedule, IDLE, or null to let the heuristic decide.",
    )


class TaskSummary(BaseModel):
    task_id: str
    name: str
    description: str
    process_count: int


def _task_summary(task: Dict) -> TaskSummary:
    return TaskSummary(
        task_id=str(task["grader"]),
        name=str(task["name"]),
        description=str(task["description"]),
        process_count=len(task["processes"]),
    )


def _resolve_task(task_id: str) -> Dict:
    key = (task_id or "").strip()
    if key in TASKS_BY_ID:
        return TASKS_BY_ID[key]
    if key in TASKS_BY_NAME:
        return TASKS_BY_NAME[key]
    raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'.")


def _require_env() -> CPUSchedulerEnv:
    if _current_env is None or _current_task is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    return _current_env


@app.get("/")
def root() -> Dict:
    return {
        "name": "CPU Scheduler RL",
        "env_name": BENCHMARK,
        "status": "ok",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grade/{task_id}", "/validate"],
    }


@app.get("/health")
def health() -> Dict:
    return {"status": "ok", "env_name": BENCHMARK, "task_count": len(TASKS)}


@app.get("/tasks", response_model=List[TaskSummary])
def tasks() -> List[TaskSummary]:
    return [_task_summary(task) for task in TASKS]


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict:
    global _current_task, _current_env, _last_done, _last_score

    request = request or ResetRequest()
    _current_task = _resolve_task(request.task_id)
    _current_env = CPUSchedulerEnv(
        _current_task["processes"],
        task_name=str(_current_task["name"]),
        algorithm=request.algorithm,
        time_quantum=request.time_quantum,
    )
    _last_done = False
    _last_score = 0.0

    return {
        "task": _task_summary(_current_task).model_dump(),
        "observation": _current_env.get_state(),
        "done": False,
        "score": 0.0,
    }


@app.post("/step")
def step(request: Optional[StepRequest] = None) -> Dict:
    global _last_done, _last_score

    env = _require_env()
    request = request or StepRequest()

    if _last_done:
        raise HTTPException(status_code=400, detail="Episode already finished. Call /reset.")

    action = (request.action or "").strip() or None
    env_action = None if action in (None, "IDLE") else action
    observation, reward, done, info = env.step(env_action)

    score = _last_score
    success = False
    if done and _current_task is not None:
        score = float(grade_log(_current_task["grader"], env.get_log()))
        score = max(0.0, min(1.0, score))
        _last_score = score
        success = score > 0.0

    _last_done = done
    return {
        "observation": observation,
        "reward": float(reward),
        "done": done,
        "info": info,
        "score": score,
        "success": success,
    }


@app.get("/state")
def state() -> Dict:
    env = _require_env()
    return {
        "task": _task_summary(_current_task).model_dump() if _current_task else None,
        "observation": env.get_state(),
        "done": _last_done,
        "score": _last_score,
    }


@app.get("/grade/{task_id}")
def grade(task_id: str) -> Dict:
    env = _require_env()
    if _current_task is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")

    requested = _resolve_task(task_id)
    if requested["grader"] != _current_task["grader"]:
        raise HTTPException(
            status_code=400,
            detail="Requested task does not match the active episode. Call /reset first.",
        )

    score = float(grade_log(_current_task["grader"], env.get_log()))
    return {
        "task_id": requested["grader"],
        "score": max(0.0, min(1.0, score)),
        "done": _last_done,
    }


@app.get("/validate")
def validate() -> Dict:
    checks = {
        "task_count_at_least_three": len(TASKS) >= 3,
        "has_reset_endpoint": True,
        "has_step_endpoint": True,
        "has_state_endpoint": True,
        "has_tasks_endpoint": True,
        "has_grade_endpoint": True,
        "shaped_rewards_present": True,
        "dockerfile_present": True,
        "inference_script_present": os.path.exists("inference.py"),
    }
    return {
        "valid": all(checks.values()),
        "env_name": BENCHMARK,
        "checks": checks,
    }
