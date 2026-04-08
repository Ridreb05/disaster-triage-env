# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Task 03 — Cascading Hazard Response (Hard)

Objective: Manage a 10-zone disaster with cascading hazard events,
blocked zones, and strict resource constraints. The agent must
contain spreading hazards while triaging injured zones.

Grading (deterministic breakdown):
  - Zone resolution rate     : 0.35 (10 zones, 0.035 each)
  - Hazard containment       : 0.25 (stop hazards from spreading to new zones)
  - Population saved ratio   : 0.20 (population_at_risk of resolved zones / total)
  - Resource efficiency      : 0.10 (no deploys to blocked/resolved zones)
  - Speed bonus              : 0.10 (≥7 zones resolved with ≥8 steps remaining)
"""

from __future__ import annotations

from simulation.incident_generator import ScenarioConfig
from tasks.base_task import BaseTask, GraderResult


class Task03CascadingHazards(BaseTask):
    task_id = "task_03_cascading_hazards"
    description = (
        "Respond to a 10-zone disaster with cascading hazards, blocked roads, "
        "and resource constraints. Contain spreading hazards while triaging "
        "the highest-severity zones. 30 steps."
    )
    difficulty = "hard"
    scenario_config = ScenarioConfig(
        num_zones=10,
        max_steps=30,
        base_resources={
            "medical_team": 5,
            "rescue_unit": 5,
            "supply_drop": 6,
            "hazmat_crew": 3,
        },
        hazard_probability=0.5,
        cascade_spread_probability=0.35,
        blocked_zone_probability=0.3,
        min_severity=0.3,
        max_severity=1.0,
        population_range=(100, 3000),
    )

    def grade(self, episode_history: list[dict]) -> GraderResult:
        if not episode_history:
            return GraderResult(
                task_id=self.task_id, score=0.0,
                feedback="No steps taken.", breakdown={}
            )

        initial_obs = episode_history[0]["observation"]
        final_obs = episode_history[-1]["observation"]
        max_steps = initial_obs.get("max_steps", 30)
        elapsed = final_obs.get("elapsed_steps", len(episode_history))

        initial_zones = {z["zone_id"]: z for z in initial_obs["zones"]}
        final_zones = {z["zone_id"]: z for z in final_obs["zones"]}
        num_zones = len(initial_zones)

        breakdown: dict[str, float] = {}

        # --- Zone resolution rate (0.35) ---
        resolved_count = sum(1 for z in final_zones.values() if z.get("resolved", False))
        resolution_score = (resolved_count / num_zones) * 0.35
        breakdown["resolution_rate"] = round(resolution_score, 4)

        # --- Hazard containment (0.25) ---
        # Proxy: measure whether initial high-hazard zones were acted on before cascade
        initial_hazard_zones = {
            zid for zid, z in initial_zones.items()
            if z.get("hazard_type", "none") not in ("none",)
        }
        hazard_zones_resolved = sum(
            1 for zid in initial_hazard_zones
            if final_zones.get(zid, {}).get("resolved", False)
        )
        containment_score = (
            hazard_zones_resolved / max(len(initial_hazard_zones), 1)
        ) * 0.25
        breakdown["hazard_containment"] = round(containment_score, 4)

        # --- Population saved ratio (0.20) ---
        total_pop = sum(z["population_at_risk"] for z in initial_zones.values())
        saved_pop = sum(
            initial_zones[zid]["population_at_risk"]
            for zid, z in final_zones.items()
            if z.get("resolved", False)
        )
        pop_score = (saved_pop / max(total_pop, 1)) * 0.20
        breakdown["population_saved"] = round(pop_score, 4)

        # --- Resource efficiency (0.10) ---
        invalid_actions = sum(
            1 for step in episode_history
            if _is_invalid_action(step, initial_zones, final_zones)
        )
        efficiency_score = max(0.0, 0.10 - invalid_actions * 0.02)
        breakdown["resource_efficiency"] = round(efficiency_score, 4)

        # --- Speed bonus (0.10) ---
        steps_remaining = max_steps - elapsed
        speed_score = 0.10 if (resolved_count >= 7 and steps_remaining >= 8) else 0.0
        breakdown["speed_bonus"] = speed_score

        total = round(
            resolution_score + containment_score + pop_score +
            efficiency_score + speed_score, 4
        )
        total = min(1.0, total)
        breakdown["total"] = total

        return GraderResult(
            task_id=self.task_id,
            score=total,
            breakdown=breakdown,
            feedback=_feedback(total, resolved_count, num_zones),
            passed=total >= 0.5,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_invalid_action(
    step: dict,
    initial_zones: dict,
    final_zones: dict,
) -> bool:
    zone_id = step["action"]["zone_id"]
    initial = initial_zones.get(zone_id, {})
    # Invalid: deploying to a zone that started as blocked
    if not initial.get("is_accessible", True):
        return True
    # Invalid: deploying to a zone that was already resolved at episode end
    # and the action happened in the last 30% of steps
    elapsed = step.get("step", 0)
    if final_zones.get(zone_id, {}).get("resolved", False) and elapsed > 20:
        return True
    return False


def _feedback(score: float, resolved: int, total: int) -> str:
    if score >= 0.85:
        return f"Outstanding: {resolved}/{total} zones resolved with hazards contained."
    elif score >= 0.70:
        return f"Strong: {resolved}/{total} zones resolved. Minor efficiency gaps."
    elif score >= 0.50:
        return f"Passing: {resolved}/{total} resolved. Hazard containment needs work."
    elif score >= 0.30:
        return f"Below passing: {resolved}/{total} zones resolved. Focus on hazard zones first."
    else:
        return f"Poor: only {resolved}/{total} zones resolved. Prioritize by severity and hazard type."
