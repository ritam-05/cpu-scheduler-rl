"""LLM policy agent for CPU scheduling decisions."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

from openai import OpenAI


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


class LLMPolicyAgent:
    """OpenAI-compatible LLM policy with retries, validation, and fallback."""

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 30) -> None:
        self.api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).strip()
        self.model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()
        self.api_key = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip()
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self.client: Optional[OpenAI] = None
        if self.api_base_url and self.model_name and self.api_key:
            self.client = OpenAI(
                base_url=self.api_base_url,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )

    @staticmethod
    def _fallback_sjf(state: Dict) -> Tuple[str, str]:
        queue = state.get("queue", [])
        if not queue:
            return "IDLE", "No ready process; CPU should stay idle."
        chosen = min(
            queue,
            key=lambda p: (
                int(p.get("remaining_time", p.get("burst_time", 0))),
                int(p.get("priority", 9999)),
                int(p.get("arrival_time", 0)),
                str(p.get("pid", "")),
            ),
        )
        return str(chosen["pid"]), "Fallback SJF selected shortest available job."

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict]:
        content = (content or "").strip()
        if not content:
            return None

        # Direct JSON parse first.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Best-effort extraction of a JSON object substring.
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = content[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def _build_messages(self, state: Dict) -> list:
        queue = state.get("queue", [])
        candidate_pids = [str(p["pid"]) for p in queue]

        system_prompt = (
            "You are an RL policy for a CPU scheduler. "
            "Goal: minimize waiting time and idle CPU time, maximize completed tasks. "
            "You must return STRICT JSON only with keys: action, reasoning. "
            "action must be one process id from available queue, or IDLE if queue empty."
        )

        few_shot = (
            "Example 1\n"
            "State queue: [{pid:P1, burst:3, priority:2}, {pid:P2, burst:1, priority:3}]\n"
            'Good output: {"action":"P2","reasoning":"Shortest job first reduces waiting."}\n\n'
            "Example 2\n"
            "State queue: [{pid:P3, burst:7, priority:1}, {pid:P4, burst:2, priority:3}]\n"
            'Good output: {"action":"P3","reasoning":"High priority task should run to avoid priority inversion."}\n'
        )

        user_prompt = (
            "Current scheduler state:\n"
            f"{json.dumps(state, separators=(',', ':'))}\n\n"
            f"Available process ids: {candidate_pids}\n"
            "Objective: optimize average waiting time, turnaround time, and CPU utilization.\n"
            "Return STRICT JSON only in the form:\n"
            '{"action":"<process_id_or_IDLE>","reasoning":"<short explanation>"}\n\n'
            f"{few_shot}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def select_action(self, state: Dict) -> Tuple[str, str, bool]:
        """Return (action, reasoning, fallback_used)."""
        queue = state.get("queue", [])
        valid_pids = {str(p["pid"]) for p in queue}

        if not queue:
            return "IDLE", "Queue empty; keeping CPU idle.", False

        # If client is unavailable, use fallback immediately.
        if self.client is None:
            action, reason = self._fallback_sjf(state)
            return action, f"{reason} LLM disabled or missing API env vars.", True

        messages = self._build_messages(state)
        backoff = 1.0

        for _attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=120,
                )
                raw = response.choices[0].message.content or ""
                parsed = self._extract_json(raw)
                if parsed is None:
                    raise ValueError("LLM output is not valid JSON")

                action = str(parsed.get("action", "")).strip()
                reasoning = (
                    str(parsed.get("reasoning", "")).strip() or "No reasoning provided."
                )
                if action not in valid_pids:
                    raise ValueError("Action not in available process IDs")

                return action, reasoning, False
            except Exception:
                time.sleep(backoff)
                backoff *= 2

        action, reason = self._fallback_sjf(state)
        return action, f"{reason} LLM call/validation failed after retries.", True
