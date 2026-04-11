from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.env import SCENARIOS, TrafficEnv
from app.models import model_to_dict
from app.tasks import TrafficTasks
import uvicorn

app = FastAPI(
    title="Traffic Optimization OpenEnv",
    version="2.0.0",
    description="Seedable four-way traffic signal environment with adaptive-control benchmarks.",
)
env = TrafficEnv()
tasks = TrafficTasks()


class ActionInput(BaseModel):
    action: int


class ResetInput(BaseModel):
    difficulty: str = "medium"
    seed: Optional[int] = None


@app.post("/reset")
def reset(input: Optional[ResetInput] = None):
    payload = input or ResetInput()
    try:
        state = env.reset(difficulty=payload.difficulty, seed=payload.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": model_to_dict(state),
        "info": env.metrics(),
    }


@app.post("/step")
def step(input: ActionInput):
    try:
        state, reward, done, info = env.step(input.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": model_to_dict(state),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


@app.get("/")
def root():
    return {
        "status": "ready",
        "project": "Traffic Optimization OpenEnv",
        "actions": {"0": "keep current signal", "1": "switch signal"},
        "difficulties": list(SCENARIOS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/tasks/{difficulty}"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/state")
def state():
    return {
        "observation": model_to_dict(env.state()),
        "info": env.metrics(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": name,
                "max_steps": scenario.max_steps,
                "service_rate": scenario.service_rate,
                "initial_queues": dict(scenario.initial_queues),
            }
            for name, scenario in SCENARIOS.items()
        ]
    }


@app.get("/tasks/{difficulty}")
def evaluate_task(difficulty: str):
    try:
        return tasks.evaluate_task(difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
