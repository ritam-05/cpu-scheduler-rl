from __future__ import annotations

import unittest

from env import CPUSchedulerEnv
from grader import grade_log
from tasks import load_tasks


class CPUSchedulerEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simple_processes = [
            {"pid": "P1", "arrival_time": 0, "burst_time": 5, "priority": 3},
            {"pid": "P2", "arrival_time": 0, "burst_time": 2, "priority": 1},
            {"pid": "P3", "arrival_time": 1, "burst_time": 1, "priority": 2},
        ]

    def test_reset_exposes_queue_state(self) -> None:
        task = load_tasks()[0]
        env = CPUSchedulerEnv(task["processes"], task_name=task["name"])

        state = env.reset()

        self.assertEqual(state["task"], task["name"])
        self.assertIn("queue", state)
        self.assertGreaterEqual(len(state["queue"]), 1)

    def test_invalid_action_falls_back_to_heuristic(self) -> None:
        task = load_tasks()[0]
        env = CPUSchedulerEnv(task["processes"], task_name=task["name"])
        env.reset()

        _state, _reward, done, info = env.step("NOT_A_PID")

        self.assertFalse(done)
        self.assertTrue(info["fallback_used"])
        self.assertEqual(info["event"], "completed")

    def test_fcfs_prefers_first_arrival(self) -> None:
        env = CPUSchedulerEnv(self.simple_processes, algorithm="fcfs")
        env.reset()

        _state, _reward, _done, info = env.step(None)

        self.assertEqual(info["selected_pid"], "P1")
        self.assertEqual(info["event"], "completed")

    def test_priority_prefers_lowest_priority_value(self) -> None:
        env = CPUSchedulerEnv(self.simple_processes, algorithm="priority")
        env.reset()

        _state, _reward, _done, info = env.step(None)

        self.assertEqual(info["selected_pid"], "P2")
        self.assertEqual(info["event"], "completed")

    def test_srjf_can_preempt_longer_job(self) -> None:
        env = CPUSchedulerEnv(self.simple_processes, algorithm="srjf")
        env.reset()

        _state, _reward, _done, info = env.step("P1")

        self.assertEqual(info["selected_pid"], "P1")
        self.assertEqual(info["event"], "preempted")
        state = env.get_state()
        self.assertEqual(state["queue"][0]["pid"], "P3")

    def test_round_robin_rotates_unfinished_work(self) -> None:
        env = CPUSchedulerEnv(self.simple_processes, algorithm="rr", time_quantum=2)
        env.reset()

        _state, _reward, _done, info = env.step(None)

        self.assertEqual(info["selected_pid"], "P1")
        self.assertEqual(info["event"], "preempted")
        state = env.get_state()
        self.assertEqual(state["queue"][0]["pid"], "P2")

    def test_all_tasks_grade_to_unit_interval(self) -> None:
        for task in load_tasks():
            env = CPUSchedulerEnv(task["processes"], task_name=task["name"])
            state = env.reset()
            done = False

            while not done:
                queue = state["queue"]
                action = queue[0]["pid"] if queue else None
                state, _reward, done, _info = env.step(action)

            score = grade_log(task["grader"], env.get_log())
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
