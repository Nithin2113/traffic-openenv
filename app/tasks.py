from typing import Callable, Dict, Tuple

from app.agent import choose_action
from app.env import SCENARIOS, TrafficEnv


TASK_SEEDS = {
    "easy": 11,
    "medium": 23,
    "hard": 47,
}


def fixed_time_action(state, step_index: int, cycle_length: int = 4) -> int:
    """Baseline controller that alternates on a fixed timer."""

    target_signal = "NS" if (step_index // cycle_length) % 2 == 0 else "EW"
    return 0 if state.signal == target_signal else 1


def adaptive_action(state, step_index: int) -> int:
    return choose_action(state)


def evaluate_policy(
    difficulty: str,
    policy: Callable,
    seed: int = None,
) -> Tuple[float, Dict]:
    if difficulty not in SCENARIOS:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(SCENARIOS)}.")

    env = TrafficEnv(difficulty=difficulty, seed=seed)
    state = env.reset(seed=seed)
    total_reward = 0.0
    info = env.metrics()

    for step_index in range(env.scenario.max_steps):
        action = policy(state, step_index)
        state, reward, done, info = env.step(action)
        total_reward += reward.value
        if done:
            break

    return round(total_reward, 2), info


def grade(score, baseline=None):
    """Normalize a raw reward into a judge-friendly 0..1 score."""

    if baseline is None:
        normalized = (score + 750) / 900
    else:
        improvement = score - baseline
        expected_gain = max(1.0, abs(baseline) * 0.25)
        normalized = 0.5 + improvement / (2 * expected_gain)

    epsilon = 1e-12
    normalized = max(epsilon, min(1.0 - epsilon, normalized))
    return normalized


class TrafficTasks:
    def run_task(self, difficulty="easy"):
        return self.evaluate_task(difficulty)["score"]

    def evaluate_task(self, difficulty="easy"):
        seed = TASK_SEEDS.get(difficulty, 0)
        adaptive_reward, adaptive_metrics = evaluate_policy(difficulty, adaptive_action, seed=seed)
        baseline_reward, baseline_metrics = evaluate_policy(difficulty, fixed_time_action, seed=seed)
        score = grade(adaptive_reward, baseline_reward)

        return {
            "difficulty": difficulty,
            "score": score,
            "adaptive_reward": adaptive_reward,
            "baseline_reward": baseline_reward,
            "reward_improvement": round(adaptive_reward - baseline_reward, 2),
            "adaptive_metrics": adaptive_metrics,
            "baseline_metrics": baseline_metrics,
        }
