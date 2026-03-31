from app.env import TrafficEnv
import time


def run_task():
    env = TrafficEnv()
    total_reward = 0

    state = env.reset()

    for _ in range(10):
        action = 1

        state, reward, done, _ = env.step(action)

        total_reward += reward

        if done:
            break

    return total_reward


def main():
    print("🚦 Traffic OpenEnv running...")

    while True:
        for level in ["easy", "medium", "hard"]:
            score = run_task()
            print(f"{level} score: {score}")

        time.sleep(5)  # 🔥 IMPORTANT: prevents CPU overload


if __name__ == "__main__":
    main()