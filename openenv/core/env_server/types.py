from __future__ import annotations

from pydantic import BaseModel, Field


class Action(BaseModel):
    """Base action type (Pydantic v2)."""


class Observation(BaseModel):
    """Base observation type (Pydantic v2)."""


class State(BaseModel):
    """Base state type (Pydantic v2)."""

    episode_id: str = Field(default="")
    step_count: int = Field(default=0, ge=0)

