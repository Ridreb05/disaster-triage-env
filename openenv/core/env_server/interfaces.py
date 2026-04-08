from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """Minimal environment interface used by this repo."""

    @abstractmethod
    def reset(self, seed: int | None = None, task_id: str | None = None) -> Any: ...

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]: ...

    @abstractmethod
    def state(self) -> Any: ...

