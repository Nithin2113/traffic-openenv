import os
from typing import Optional

from openai import OpenAI

from app.agent import choose_action, reset_agent
from app.env import TrafficEnv


TASKS = ["easy", "medium", "hard"]
TASK_SEEDS = {
    "easy": 11,
    "medium": 23,
    "hard": 47,
}


def build_client() -> Optional[OpenAI]:
    """Create an OpenAI-compatible client from hackathon env variables."""

    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        return None

    try:
        return OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=api_key,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "2.5")),
        )
    except Exception:
        return None


def observation_to_dict(state):
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state.dict()


def call_llm_for_action(client: Optional[OpenAI], state) -> Optional[int]:
    """Ask the provided LLM endpoint for an action every step when available."""

    if client is None:
        return None

    prompt = (
        "You are controlling a two-phase traffic signal. "
        "Action 0 means keep the current signal. "
        "Action 1 means switch to the other signal. "
        f"Observation: {observation_to_dict(state)}. "
        "Return only one character: 0 or 1."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1,
        )
        content = response.choices[0].message.content.strip()
    except Exception:
        return None

    if content in {"0", "1"}:
        return int(content)

    return None


def score_from_reward(total_reward: float) -> float:
    """Hackathon-required score: strictly between 0 and 1."""

    score = 1 / (1 + abs(total_reward))
    return clamp_open_interval(score)


def clamp_open_interval(value: float, epsilon: float = 1e-12) -> float:
    """Clamp a numeric value into the open interval (0, 1)."""

    if value <= 0.0:
        return epsilon
    if value >= 1.0:
        return 1.0 - epsilon
    return value


def normalize_step_reward(raw_reward: float) -> float:
    """Map raw environment reward into (0, 1) while preserving sign information.

    Positive rewards stay above 0.5, negative rewards stay below 0.5, and the
    output never reaches exactly 0 or 1.
    """

    scale = 25.0
    normalized = 0.5 + 0.5 * (raw_reward / (abs(raw_reward) + scale))
    return clamp_open_interval(normalized)


def format_metric(value: float, decimals: int = 12) -> str:
    """Return a richer numeric string without collapsing to 2 decimal places."""

    return f"{value:.{decimals}f}"


def run_task(task_name: str, client: Optional[OpenAI]) -> float:
    print(f"[START] task={task_name}", flush=True)

    env = TrafficEnv(difficulty=task_name, seed=TASK_SEEDS.get(task_name, 0))
    state = env.reset(seed=TASK_SEEDS.get(task_name, 0))
    reset_agent()

    total_reward = 0.0
    steps = 0

    for step_number in range(1, env.scenario.max_steps + 1):
        action = call_llm_for_action(client, state)
        if action is None:
            action = choose_action(state)

        state, reward, done, _ = env.step(action)
        raw_reward_value = float(getattr(reward, "value", reward))
        logged_reward_value = normalize_step_reward(raw_reward_value)
        total_reward += raw_reward_value
        steps += 1

        print(
            f"[STEP] step={step_number} reward={format_metric(logged_reward_value)}",
            flush=True,
        )

        if done:
            break

    score = score_from_reward(total_reward)
    print(f"[END] task={task_name} score={format_metric(score)} steps={steps}", flush=True)
    return score


def main():
    client = build_client()
    for task_name in TASKS:
        run_task(task_name, client)


if __name__ == "__main__":
    main()
