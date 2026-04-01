from app.env import TrafficEnv
import time


def run_task(level: str):
    env = TrafficEnv()
    total_reward = 0

    state = env.reset()

    for _ in range(10):
        action = 1  # simple policy

        state, reward, done, _ = env.step(action)

        # handle Reward object safely
        total_reward += getattr(reward, "value", reward)

        if done:
            break

    return total_reward


def main():
    print("🚦 Traffic OpenEnv running...")

    while True:
        for level in ["easy", "medium", "hard"]:
            score = run_task(level)
            print(f"{level} score: {score}")

        time.sleep(5)  # prevents CPU overload


if __name__ == "__main__":
    main()