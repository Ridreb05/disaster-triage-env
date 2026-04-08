# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Task 02 — Multi-Zone Priority Routing (Medium)

Objective: Triage 5 simultaneous incident zones with limited resources.
The agent must prioritize zones by severity and deploy appropriate
resources to resolve at least 3 of 5 zones within 15 steps.

Grading (deterministic breakdown):
  - Zone resolution rate   : 0.50 (0.10 per zone, max 5)
  - Priority ordering score: 0.25 (were high-severity zones addressed first?)
  - Resource efficiency    : 0.15 (no wasted deploys to resolved zones)
  - Speed bonus            : 0.10 (all 3+ zones resolved with ≥5 steps remaining)
"""

from __future__ import annotations

from simulation.incident_generator import ScenarioConfig
from tasks.base_task import BaseTask, GraderResult


class Task02MultiZone(BaseTask):
    task_id = "task_02_multi_zone"
    description = (
        "Triage 5 simultaneous incident zones with limited resources. "
        "Prioritize by severity and resolve at least 3 zones in 15 steps."
    )
    difficulty = "medium"
    scenario_config = ScenarioConfig(
        num_zones=5,
        max_steps=15,
        base_resources={
            "medical_team": 4,
            "rescue_unit": 3,
            "supply_drop": 4,
            "hazmat_crew": 2,
        },
        hazard_probability=0.2,
        cascade_spread_probability=0.1,
        blocked_zone_probability=0.1,
        min_severity=0.2,
        max_severity=1.0,
    )

    def grade(self, episode_history: list[dict]) -> GraderResult:
        if not episode_history:
            return GraderResult(
                task_id=self.task_id, score=0.0,
                feedback="No steps taken.", breakdown={}
            )

        initial_obs = episode_history[0]["observation"]
        final_obs = episode_history[-1]["observation"]
        max_steps = initial_obs.get("max_steps", 15)
        elapsed = final_obs.get("elapsed_steps", len(episode_history))

        # Sort zones by severity descending (ground truth priority order)
        initial_zones = sorted(initial_obs["zones"], key=lambda z: z["severity"], reverse=True)
        final_zones = {z["zone_id"]: z for z in final_obs["zones"]}
        breakdown: dict[str, float] = {}

        # --- Zone resolution rate (0.50) ---
        resolved_count = sum(
            1 for z in final_zones.values() if z.get("resolved", False)
        )
        resolution_score = min(1.0, resolved_count / 5) * 0.50
        breakdown["resolution_rate"] = round(resolution_score, 4)

        # --- Priority ordering score (0.25) ---
        # Check: were the top-2 severity zones acted on before the bottom-2?
        top_zones = {z["zone_id"] for z in initial_zones[:2]}
        bottom_zones = {z["zone_id"] for z in initial_zones[-2:]}

        first_top_step = _first_action_on_zones(episode_history, top_zones)
        first_bottom_step = _first_action_on_zones(episode_history, bottom_zones)

        if first_top_step is not None and (
            first_bottom_step is None or first_top_step < first_bottom_step
        ):
            priority_score = 0.25
        elif first_top_step is not None:
            priority_score = 0.10  # partial: acted on top zones but not before bottom
        else:
            priority_score = 0.0
        breakdown["priority_ordering"] = priority_score

        # --- Resource efficiency (0.15) ---
        wasted_actions = sum(
            1 for step in episode_history
            if _is_wasted_action(step, final_zones)
        )
        efficiency_score = max(0.0, 0.15 - wasted_actions * 0.03)
        breakdown["resource_efficiency"] = round(efficiency_score, 4)

        # --- Speed bonus (0.10) ---
        steps_remaining = max_steps - elapsed
        speed_score = 0.10 if (resolved_count >= 3 and steps_remaining >= 5) else 0.0
        breakdown["speed_bonus"] = speed_score

        total = round(
            resolution_score + priority_score + efficiency_score + speed_score, 4
        )
        total = min(1.0, total)
        breakdown["total"] = total

        return GraderResult(
            task_id=self.task_id,
            score=total,
            breakdown=breakdown,
            feedback=_feedback(total, resolved_count),
            passed=total >= 0.5,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_action_on_zones(history: list[dict], zone_ids: set[int]) -> int | None:
    for i, step in enumerate(history):
        if step["action"]["zone_id"] in zone_ids:
            return i
    return None


def _is_wasted_action(step: dict, final_zones: dict) -> bool:
    zone_id = step["action"]["zone_id"]
    zone = final_zones.get(zone_id, {})
    # Wasted if zone was resolved AND this action happened after it was resolved
    # (approximation: we count any action on a fully-resolved-at-end zone beyond step 1)
    return zone.get("resolved", False) and step["step"] > 1


def _feedback(score: float, resolved: int) -> str:
    if score >= 0.9:
        return f"Excellent: {resolved}/5 zones resolved with optimal priority ordering."
    elif score >= 0.7:
        return f"Good: {resolved}/5 zones resolved. Minor priority or efficiency issues."
    elif score >= 0.5:
        return f"Passing: {resolved}/5 zones resolved, but priority ordering suboptimal."
    elif resolved >= 3:
        return f"{resolved}/5 zones resolved but resource efficiency was poor."
    else:
        return f"Only {resolved}/5 zones resolved. Focus on highest-severity zones first."
