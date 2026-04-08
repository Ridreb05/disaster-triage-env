# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Task 01 — Single Zone Triage (Easy)

Objective: Identify the single highest-severity zone and deploy the
correct resource type to resolve it within 5 steps.

Grading:
  - 0.0  : No action taken, or wrong zone targeted entirely.
  - 0.3  : Correct zone targeted but wrong resource type.
  - 0.6  : Correct zone + correct resource, but not resolved.
  - 0.85 : Resolved correctly.
  - 1.0  : Resolved correctly within 3 steps (speed bonus).
"""

from __future__ import annotations

from simulation.incident_generator import ScenarioConfig
from tasks.base_task import BaseTask, GraderResult


class Task01SingleZone(BaseTask):
    task_id = "task_01_single_zone"
    description = (
        "Identify the highest-severity incident zone and deploy the correct "
        "resource type to resolve it. One zone, one resource type, 5 steps."
    )
    difficulty = "easy"
    scenario_config = ScenarioConfig(
        num_zones=3,
        max_steps=5,
        base_resources={
            "medical_team": 3,
            "rescue_unit": 3,
            "supply_drop": 3,
            "hazmat_crew": 2,
        },
        hazard_probability=0.0,
        blocked_zone_probability=0.0,
        min_severity=0.3,
        max_severity=1.0,
    )

    def grade(self, episode_history: list[dict]) -> GraderResult:
        if not episode_history:
            return GraderResult(
                task_id=self.task_id, score=0.0,
                feedback="No steps taken.", breakdown={"no_actions": 0.0}
            )

        # Find the target zone from the initial observation
        initial_obs = episode_history[0]["observation"]
        zones = initial_obs["zones"]
        target_zone = max(zones, key=lambda z: z["severity"])
        target_zone_id = target_zone["zone_id"]

        breakdown: dict[str, float] = {}
        score = 0.0

        # Check if agent ever targeted the correct zone
        correct_zone_actions = [
            step for step in episode_history
            if step["action"]["zone_id"] == target_zone_id
        ]

        if not correct_zone_actions:
            breakdown["correct_zone"] = 0.0
            return GraderResult(
                task_id=self.task_id, score=0.0,
                feedback=f"Never targeted highest-severity zone {target_zone_id}.",
                breakdown=breakdown
            )

        score += 0.3
        breakdown["correct_zone"] = 0.3

        # Check resource affinity
        from simulation.hazard_physics import resolve_zone
        from models import ResourceType, ZoneState, HazardType

        # Determine optimal resource for the target zone's hazard
        hazard = target_zone.get("hazard_type", "none")
        optimal_resources = _optimal_resource_for_hazard(hazard)

        correct_resource_actions = [
            step for step in correct_zone_actions
            if step["action"]["resource_type"] in optimal_resources
        ]

        if correct_resource_actions:
            score += 0.3
            breakdown["correct_resource"] = 0.3
        else:
            breakdown["correct_resource"] = 0.0

        # Check if zone was resolved
        final_obs = episode_history[-1]["observation"]
        final_zones = {z["zone_id"]: z for z in final_obs["zones"]}
        target_resolved = final_zones.get(target_zone_id, {}).get("resolved", False)

        if target_resolved:
            score += 0.25
            breakdown["zone_resolved"] = 0.25

            # Speed bonus: resolved within first 3 steps
            first_resolve_step = next(
                (i for i, step in enumerate(episode_history)
                 if step["observation"]["zones"][target_zone_id].get("resolved", False)),
                len(episode_history)
            )
            if first_resolve_step <= 2:
                score += 0.15
                breakdown["speed_bonus"] = 0.15
        else:
            breakdown["zone_resolved"] = 0.0

        score = round(min(1.0, score), 4)
        passed = score >= 0.5
        return GraderResult(
            task_id=self.task_id,
            score=score,
            breakdown=breakdown,
            feedback=_feedback(score),
            passed=passed,
        )


def _optimal_resource_for_hazard(hazard: str) -> list[str]:
    return {
        "earthquake": ["medical_team", "rescue_unit"],
        "flood":      ["rescue_unit", "supply_drop"],
        "hazmat":     ["hazmat_crew"],
        "fire":       ["medical_team", "hazmat_crew"],
        "none":       ["medical_team", "rescue_unit", "supply_drop"],
    }.get(hazard, ["medical_team", "rescue_unit"])


def _feedback(score: float) -> str:
    if score >= 1.0:
        return "Perfect: correct zone, correct resource, resolved under 3 steps."
    elif score >= 0.85:
        return "Zone resolved with correct resource. Speed bonus missed."
    elif score >= 0.6:
        return "Correct zone and resource deployed, but zone not fully resolved."
    elif score >= 0.3:
        return "Correct zone targeted but wrong resource type for the hazard."
    else:
        return "Highest-severity zone never targeted."
