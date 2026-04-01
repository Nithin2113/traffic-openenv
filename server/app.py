from fastapi import FastAPI
from pydantic import BaseModel
from app.env import TrafficEnv

app = FastAPI()

env = TrafficEnv()


class ActionInput(BaseModel):
    action: int


@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "observation": getattr(state, "dict", lambda: state)()
    }


@app.post("/step")
def step(input: ActionInput):
    state, reward, done, _ = env.step(input.action)

    return {
        "observation": getattr(state, "dict", lambda: state)(),
        "reward": getattr(reward, "value", reward),
        "done": done,
        "info": {}
    }


@app.get("/")
def root():
    return {"status": "ok"}