# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Abstract base task. All tasks must implement grade() deterministically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from simulation.incident_generator import ScenarioConfig


def clamp_score(score: float) -> float:
    """Clamp score to the open interval (0.01, 0.99).

    The OpenEnv validator rejects scores exactly equal to 0.0 or 1.0
    (requires strictly between 0 and 1). This utility ensures every
    GraderResult is compliant regardless of raw arithmetic outcomes.
    """
    return round(max(0.01, min(0.99, score)), 4)


@dataclass
class GraderResult:
    """Structured grading output returned by /grader endpoint."""
    task_id: str
    score: float                          # strictly in (0.01, 0.99) after __post_init__
    max_score: float = 1.0
    breakdown: dict = field(default_factory=dict)
    feedback: str = ""
    passed: bool = False

    def __post_init__(self):
        # Enforce the open-interval requirement imposed by the OpenEnv validator.
        # Scores of exactly 0.0 or 1.0 cause disqualification.
        self.score = clamp_score(self.score)
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
            GraderResult with score strictly in (0.01, 0.99)
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
