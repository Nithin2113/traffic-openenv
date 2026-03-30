import requests

BASE_URL = "http://127.0.0.1:8000"

def run_task():
    total_reward = 0

    # reset env
    state = requests.post(f"{BASE_URL}/reset").json()

    for _ in range(10):
        action = {"action": 1}  # simple agent

        response = requests.post(f"{BASE_URL}/step", json=action).json()

        total_reward += response["reward"]

        if response["done"]:
            break

    return total_reward


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        score = run_task()
        print(level, "score:", score)