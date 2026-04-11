from app.agent import choose_action
from app.env import TrafficEnv


def run_demo():
    env = TrafficEnv(difficulty="medium", seed=7)
    state = env.reset()
    print("Initial:", state)

    for i in range(5):
        action = choose_action(state)
        state, reward, done, info = env.step(action)
        print(f"Step {i + 1}:", state, reward, info)
        if done:
            break


if __name__ == "__main__":
    run_demo()
