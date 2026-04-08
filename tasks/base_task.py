# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Abstract base task. All tasks must implement grade() deterministically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from simulation.incident_generator import ScenarioConfig


@dataclass
class GraderResult:
    """Structured grading output returned by /grader endpoint."""
    task_id: str
    score: float                          # 0.0 – 1.0
    max_score: float = 1.0
    breakdown: dict = field(default_factory=dict)
    feedback: str = ""
    passed: bool = False

    def __post_init__(self):
        assert 0.0 <= self.score <= 1.0, f"Score {self.score} out of range"
        self.passed = self.score >= self.passing_threshold

    @property
    def passing_threshold(self) -> float:
        return 0.5


class BaseTask(ABC):
    """
    All tasks must subclass this and implement:
      - task_id: str
      - description: str
      - scenario_config: ScenarioConfig
      - grade(episode_history) -> GraderResult
    """

    task_id: str = ""
    description: str = ""
    difficulty: str = "easy"
    scenario_config: ScenarioConfig = ScenarioConfig()

    @abstractmethod
    def grade(self, episode_history: list[dict]) -> GraderResult:
        """
        Deterministically grade a completed episode.

        Args:
            episode_history: List of step dicts, each containing:
                {
                  "step": int,
                  "action": DisasterAction.model_dump(),
                  "observation": DisasterObservation.model_dump(),
                  "reward": float,
                  "done": bool,
                }

        Returns:
            GraderResult with score in [0.0, 1.0]
        """
        ...

    def action_schema(self) -> dict:
        """Returns the JSON schema for the action space of this task."""
        from models import DisasterAction
        return DisasterAction.model_json_schema()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "difficulty": self.difficulty,
            "num_zones": self.scenario_config.num_zones,
            "max_steps": self.scenario_config.max_steps,
            "action_schema": self.action_schema(),
        }
