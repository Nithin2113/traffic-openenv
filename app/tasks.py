from app.env import TrafficEnv

class TrafficTasks:
    def __init__(self):
        self.env = TrafficEnv()

    def run_task(self, difficulty="easy"):
        state = self.env.reset()

        # difficulty setup
        if difficulty == "easy":
            self.env.state_data.update({"north": 2, "south": 2, "east": 2, "west": 2})

        elif difficulty == "medium":
            self.env.state_data.update({"north": 5, "south": 5, "east": 5, "west": 5})

        elif difficulty == "hard":
            self.env.state_data.update({"north": 10, "south": 10, "east": 10, "west": 10})

        total_reward = 0

        for _ in range(10):
            state, reward, done, _ = self.env.step(1)
            total_reward += reward.value
            if done:
                break

        return total_reward


def grade(score):
    normalized = (score + 600) / 600
    return max(0, min(1, normalized))