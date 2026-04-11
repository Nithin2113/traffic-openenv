import random
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from app.models import Observation, Reward


DIRECTIONS = ("north", "south", "east", "west")
AXES = {
    "NS": ("north", "south"),
    "EW": ("east", "west"),
}
NEXT_SIGNAL = {"NS": "EW", "EW": "NS"}


@dataclass(frozen=True)
class TrafficScenario:
    name: str
    initial_queues: Mapping[str, int]
    arrival_bounds: Mapping[str, Tuple[int, int]]
    max_steps: int
    service_rate: int
    switch_loss: int = 1


SCENARIOS: Dict[str, TrafficScenario] = {
    "easy": TrafficScenario(
        name="easy",
        initial_queues={"north": 3, "south": 2, "east": 2, "west": 3},
        arrival_bounds={
            "north": (0, 1),
            "south": (0, 1),
            "east": (0, 1),
            "west": (0, 1),
        },
        max_steps=20,
        service_rate=3,
    ),
    "medium": TrafficScenario(
        name="medium",
        initial_queues={"north": 7, "south": 6, "east": 5, "west": 8},
        arrival_bounds={
            "north": (0, 2),
            "south": (0, 2),
            "east": (1, 2),
            "west": (0, 2),
        },
        max_steps=24,
        service_rate=3,
    ),
    "hard": TrafficScenario(
        name="hard",
        initial_queues={"north": 14, "south": 12, "east": 11, "west": 15},
        arrival_bounds={
            "north": (1, 3),
            "south": (1, 3),
            "east": (1, 3),
            "west": (1, 3),
        },
        max_steps=30,
        service_rate=4,
    ),
}


class TrafficEnv:
    """A deterministic, seedable traffic-signal environment."""

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        if difficulty not in SCENARIOS:
            raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(SCENARIOS)}.")

        self.rng = random.Random(seed)
        self.seed = seed
        self.scenario = SCENARIOS[difficulty]
        self.state_data = None
        self.step_count = 0
        self.phase_age = 0
        self.prev_total_queue = 0
        self.cumulative_queue = 0
        self.total_throughput = 0
        self.switch_count = 0
        self.last_info = {}

    def reset(self, difficulty: Optional[str] = None, seed: Optional[int] = None):
        if difficulty is not None:
            if difficulty not in SCENARIOS:
                raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(SCENARIOS)}.")
            self.scenario = SCENARIOS[difficulty]

        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)

        self.state_data = {direction: self.scenario.initial_queues[direction] for direction in DIRECTIONS}
        self.state_data["signal"] = "NS"
        self.step_count = 0
        self.phase_age = 0
        self.prev_total_queue = self._total_queue()
        self.cumulative_queue = 0
        self.total_throughput = 0
        self.switch_count = 0
        self.last_info = {
            "step": 0,
            "scenario": self.scenario.name,
            "signal": self.state_data["signal"],
            "switched": False,
            "arrivals": {direction: 0 for direction in DIRECTIONS},
            "departed": {direction: 0 for direction in DIRECTIONS},
            "throughput": 0,
            "total_throughput": 0,
            "total_queue": self.prev_total_queue,
            "average_queue": 0.0,
            "switches": 0,
            "phase_age": 0,
        }

        return self._observation()

    def step(self, action: int) -> Tuple[Observation, Reward, bool, dict]:
        if self.state_data is None:
            self.reset()

        if action not in (0, 1):
            raise ValueError("Action must be 0 to keep the signal or 1 to switch it.")

        self.step_count += 1

        switched = action == 1
        if switched:
            self.state_data["signal"] = NEXT_SIGNAL[self.state_data["signal"]]
            self.phase_age = 0
            self.switch_count += 1
        else:
            self.phase_age += 1

        service_rate = max(1, self.scenario.service_rate - (self.scenario.switch_loss if switched else 0))
        active_directions = AXES[self.state_data["signal"]]
        departed = {direction: 0 for direction in DIRECTIONS}

        for direction in active_directions:
            departed[direction] = min(self.state_data[direction], service_rate)
            self.state_data[direction] -= departed[direction]

        arrivals = self._arrivals()
        for direction in DIRECTIONS:
            self.state_data[direction] += arrivals[direction]

        total_queue = self._total_queue()
        throughput = sum(departed.values())
        self.cumulative_queue += total_queue
        self.total_throughput += throughput

        ns_queue = self.state_data["north"] + self.state_data["south"]
        ew_queue = self.state_data["east"] + self.state_data["west"]
        imbalance = abs(ns_queue - ew_queue)
        queue_delta = self.prev_total_queue - total_queue
        max_lane_queue = max(self.state_data[direction] for direction in DIRECTIONS)

        reward = (
            throughput * 3.0
            + queue_delta * 1.5
            - total_queue * 0.65
            - imbalance * 0.25
            - max(0, max_lane_queue - 18) * 0.75
            - (2.5 if switched else 0.0)
        )

        average_queue = self.cumulative_queue / self.step_count
        info = {
            "step": self.step_count,
            "scenario": self.scenario.name,
            "signal": self.state_data["signal"],
            "switched": switched,
            "arrivals": arrivals,
            "departed": departed,
            "throughput": throughput,
            "total_throughput": self.total_throughput,
            "total_queue": total_queue,
            "average_queue": round(average_queue, 2),
            "switches": self.switch_count,
            "phase_age": self.phase_age,
        }
        self.last_info = info
        self.prev_total_queue = total_queue

        done = self.step_count >= self.scenario.max_steps
        return self._observation(), Reward(value=round(reward, 2)), done, info

    def state(self):
        if self.state_data is None:
            self.reset()
        return self._observation()

    def metrics(self):
        return dict(self.last_info)

    def _observation(self):
        return Observation(**self.state_data, phase_age=self.phase_age)

    def _total_queue(self):
        return sum(self.state_data[direction] for direction in DIRECTIONS)

    def _arrivals(self):
        return {
            direction: self.rng.randint(*self.scenario.arrival_bounds[direction])
            for direction in DIRECTIONS
        }
