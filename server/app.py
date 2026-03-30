from fastapi import FastAPI
from pydantic import BaseModel
from app.env import TrafficEnv
import uvicorn

app = FastAPI()

env = TrafficEnv()

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


# 🔥 REQUIRED MAIN FUNCTION
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# 🔥 REQUIRED ENTRY CHECK
if __name__ == "__main__":
    main()