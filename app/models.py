from pydantic import BaseModel

# what agent sees
class Observation(BaseModel):
    north: int
    south: int
    east: int
    west: int
    signal: str


# what agent does
class Action(BaseModel):
    action: int  # 0 or 1


# reward structure
class Reward(BaseModel):
    value: float