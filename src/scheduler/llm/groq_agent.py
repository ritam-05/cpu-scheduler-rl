"""Groq-powered LLM agent for analyzing CPU scheduling results."""

import os
import json
from typing import Dict, List, Any
from groq import Groq, APIError, RateLimitError
from dotenv import load_dotenv

from scheduler.models import Process
from scheduler.metrics import MetricsReport


class SchedulingAnalyst:
    """Uses Groq's API to interpret scheduling workloads and benchmark results."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        # Explicitly load the .env file so os.getenv can find your key!
        load_dotenv()
        
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # Enable mock mode if key is missing or set to the default example template
        self.mock_mode = not self.api_key or self.api_key == "gsk_your_api_key_here"
        
        if not self.mock_mode:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def analyze_benchmark(
        self, 
        workload: List[Process], 
        results: Dict[str, MetricsReport]
    ) -> str:
        """
        Send workload and results to the LLM for analysis.
        Returns the LLM's text response.
        """
        if self.mock_mode:
            return self._get_mock_response(results)

        prompt = self._build_prompt(workload, results)

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Operating Systems Engineer. "
                            "Analyze the provided CPU scheduling benchmark. "
                            "Explain why certain algorithms outperformed others based on the workload characteristics. "
                            "Keep your response concise, professional, and limited to 3 paragraphs."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=500,
                timeout=10.0
            )
            return response.choices[0].message.content

        except RateLimitError:
            return "[Groq API Error] Rate limit exceeded. Please try again later."
        except APIError as e:
            return f"[Groq API Error] {str(e)}"
        except Exception as e:
            return f"[LLM Integration Error] {str(e)}"

    def _build_prompt(self, workload: List[Process], results: Dict[str, MetricsReport]) -> str:
        """Construct a clean, stringified representation of the data for the LLM."""
        prompt = "### WORKLOAD CHARACTERISTICS ###\n"
        for p in workload:
            prompt += f"- {p.pid}: Arrival={p.arrival_time}, Burst={p.burst_time}, Priority={p.priority}\n"
            
        prompt += "\n### BENCHMARK RESULTS ###\n"
        for algo_name, metrics in results.items():
            prompt += f"{algo_name}:\n"
            prompt += f"  Avg Wait Time: {metrics.avg_waiting_time:.2f}\n"
            prompt += f"  Avg Turnaround: {metrics.avg_turnaround_time:.2f}\n"
            prompt += f"  Context Switches: {metrics.context_switches}\n"
            
        prompt += "\nPlease analyze these results. Which algorithm performed best and why?"
        return prompt

    def _get_mock_response(self, results: Dict[str, MetricsReport]) -> str:
        """Fallback response when API key is missing to ensure the pipeline doesn't crash."""
        best_algo = min(results.items(), key=lambda x: x[1].avg_waiting_time)[0]
        
        return (
            "[MOCK MODE ANALYST REPORT]\n"
            "Groq API Key not found. Generating heuristic-based mock analysis:\n\n"
            f"Based on the provided metrics, {best_algo} achieved the lowest average waiting time. "
            "Algorithms that utilize preemption (like SRTF) typically minimize waiting time for short "
            "processes but incur higher context switching overhead. The RL agent's performance depends on "
            "its training convergence and the specific workload distribution."
        )