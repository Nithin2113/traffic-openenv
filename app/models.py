from typing import Dict, Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Traffic queues visible to the controller."""

    north: int = Field(ge=0)
    south: int = Field(ge=0)
    east: int = Field(ge=0)
    west: int = Field(ge=0)
    signal: Literal["NS", "EW"]
    phase_age: int = Field(default=0, ge=0)


class Action(BaseModel):
    """Binary traffic signal action."""

    action: int = Field(ge=0, le=1, description="0 keeps the current signal, 1 switches it")


class Reward(BaseModel):
    """Scalar reward returned by each environment step."""

    value: float


class StepInfo(BaseModel):
    """Human-readable diagnostics for judging and debugging."""

    step: int
    scenario: str
    signal: Literal["NS", "EW"]
    switched: bool
    arrivals: Dict[str, int]
    departed: Dict[str, int]
    throughput: int
    total_throughput: int
    total_queue: int
    average_queue: float
    switches: int
    phase_age: int


def model_to_dict(model):
    """Return a Pydantic model as a dict across Pydantic v1 and v2."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
