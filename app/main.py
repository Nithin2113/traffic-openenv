from fastapi import FastAPI
from pydantic import BaseModel
from app.env import TrafficEnv

app = FastAPI()

env = TrafficEnv()

# request model for step
class ActionInput(BaseModel):
    action: int


@app.post("/reset")
def reset():
    state = env.reset()
    return state.dict()


@app.post("/step")
def step(input: ActionInput):
    state, reward, done, _ = env.step(input.action)
    return {
        "state": state.dict(),
        "reward": reward.value,
        "done": done
    }


@app.get("/state")
def get_state():
    return env.state().dict()