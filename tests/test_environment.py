import unittest

from app.agent import choose_action
from app.env import TrafficEnv
from app.models import Observation
from app.tasks import TrafficTasks


class TrafficEnvironmentTests(unittest.TestCase):
    def test_seeded_runs_are_reproducible(self):
        actions = [0, 0, 1, 0, 1, 0]
        first = self._rollout(actions)
        second = self._rollout(actions)
        self.assertEqual(first, second)

    def test_invalid_action_is_rejected(self):
        env = TrafficEnv(difficulty="easy", seed=11)
        env.reset()

        with self.assertRaises(ValueError):
            env.step(2)

    def test_controller_respects_current_signal(self):
        state = Observation(north=1, south=1, east=8, west=7, signal="EW", phase_age=3)
        self.assertEqual(choose_action(state), 0)

    def test_tasks_report_adaptive_improvement(self):
        result = TrafficTasks().evaluate_task("medium")
        self.assertGreater(result["adaptive_reward"], result["baseline_reward"])
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

    def _rollout(self, actions):
        env = TrafficEnv(difficulty="medium", seed=23)
        state = env.reset(seed=23)
        rows = [state.model_dump() if hasattr(state, "model_dump") else state.dict()]

        for action in actions:
            state, reward, done, info = env.step(action)
            rows.append(
                (
                    state.model_dump() if hasattr(state, "model_dump") else state.dict(),
                    reward.value,
                    done,
                    info["total_queue"],
                    info["throughput"],
                )
            )

        return rows


if __name__ == "__main__":
    unittest.main()
