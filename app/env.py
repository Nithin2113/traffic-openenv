import random
from typing import Tuple
from app.models import Observation, Reward

class TrafficEnv:
    def __init__(self):
        self.state_data = None
        self.step_count = 0
        self.prev_wait = None

    def reset(self):
        self.state_data = {
            "north": random.randint(1, 10),
            "south": random.randint(1, 10),
            "east": random.randint(1, 10),
            "west": random.randint(1, 10),
            "signal": "NS"
        }

        self.step_count = 0

        self.prev_wait = (
            self.state_data["north"] +
            self.state_data["south"] +
            self.state_data["east"] +
            self.state_data["west"]
        )

        return Observation(**self.state_data)

    def step(self, action: int) -> Tuple[Observation, Reward, bool, dict]:
        self.step_count += 1

        switched = False
        if action == 1:
            switched = True
            if self.state_data["signal"] == "NS":
                self.state_data["signal"] = "EW"
            else:
                self.state_data["signal"] = "NS"

        # traffic flow
        if self.state_data["signal"] == "NS":
            self.state_data["north"] = max(0, self.state_data["north"] - 2)
            self.state_data["south"] = max(0, self.state_data["south"] - 2)
            self.state_data["east"] += 1
            self.state_data["west"] += 1
        else:
            self.state_data["east"] = max(0, self.state_data["east"] - 2)
            self.state_data["west"] = max(0, self.state_data["west"] - 2)
            self.state_data["north"] += 1
            self.state_data["south"] += 1

        # incoming traffic
        self.state_data["north"] += random.randint(0, 2)
        self.state_data["south"] += random.randint(0, 2)
        self.state_data["east"] += random.randint(0, 2)
        self.state_data["west"] += random.randint(0, 2)

        total_wait = (
            self.state_data["north"] +
            self.state_data["south"] +
            self.state_data["east"] +
            self.state_data["west"]
        )

        # reward logic
        reward = -total_wait

        if total_wait < self.prev_wait:
            reward += 5
        else:
            reward -= 5

        if switched:
            reward -= 2

        if total_wait < 10:
            reward += 5

        if total_wait < 5:
            reward += 10

        self.prev_wait = total_wait

        done = self.step_count >= 10

        return Observation(**self.state_data), Reward(value=reward), done, {}

    def state(self):
        return Observation(**self.state_data)