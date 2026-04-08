# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Reward calculator — provides a rich per-step signal, not just terminal reward.

Design principles:
  1. Partial progress at every step (not binary end-of-episode).
  2. Time decay: waiting costs lives — severity × time_since_incident penalty.
  3. Priority alignment: deploying to IMMEDIATE zones earns a bonus.
  4. Resource efficiency: don't over-deploy to already-resolved zones.
  5. Hazard containment bonus when a cascading event is blocked.
"""

from __future__ import annotations

from models import (
    DisasterAction, ZoneState, ResourceInventory,
    TriagePriority, ResourceType, HazardType
)


# ---------------------------------------------------------------------------
# Reward weights (tunable without changing logic)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "severity_coverage": 2.0,       # base reward per severity-unit resolved
    "population_saved": 0.001,       # per person-at-risk covered
    "time_decay_penalty": -0.05,     # per step of delay on unresolved high-severity zones
    "priority_alignment_bonus": 0.3, # bonus for correct START triage classification
    "resource_waste_penalty": -0.2,  # deploying to already-resolved zone
    "inaccessible_penalty": -0.4,    # attempting to deploy to blocked zone
    "hazard_containment_bonus": 0.5, # when hazard spread is blocked by correct resource
    "over_deploy_penalty": -0.1,     # more than 2× required resources to one zone
    "reasoning_bonus": 0.05,         # small bonus for providing reasoning field
}

# Resource → HazardType affinity (correct match earns full reward)
RESOURCE_HAZARD_AFFINITY: dict[ResourceType, list[HazardType]] = {
    ResourceType.MEDICAL_TEAM:  [HazardType.EARTHQUAKE, HazardType.FIRE],
    ResourceType.RESCUE_UNIT:   [HazardType.FLOOD, HazardType.EARTHQUAKE],
    ResourceType.SUPPLY_DROP:   [HazardType.FLOOD, HazardType.NONE],
    ResourceType.HAZMAT_CREW:   [HazardType.HAZMAT],
}


class RewardCalculator:
    """
    Stateless reward calculator. Call compute() after each step.
    Returns a float reward and a breakdown dict for debugging/info.
    """

    @staticmethod
    def compute(
        action: DisasterAction,
        zone: ZoneState,
        resources_before: ResourceInventory,
        elapsed_steps: int,
    ) -> tuple[float, dict]:
        """
        Compute the per-step reward for a single action on a single zone.

        Returns:
            reward      — scalar float
            breakdown   — dict of component rewards for transparency
        """
        breakdown: dict[str, float] = {}
        reward = 0.0

        # --- Invalid action penalties ---
        if not zone.is_accessible:
            r = WEIGHTS["inaccessible_penalty"]
            breakdown["inaccessible_penalty"] = r
            return r, breakdown

        if zone.resolved:
            r = WEIGHTS["resource_waste_penalty"]
            breakdown["resource_waste_penalty"] = r
            return r, breakdown

        # --- Core coverage reward ---
        coverage = zone.severity * WEIGHTS["severity_coverage"]
        breakdown["severity_coverage"] = coverage
        reward += coverage

        # --- Population at risk bonus ---
        pop_reward = zone.population_at_risk * WEIGHTS["population_saved"]
        breakdown["population_saved"] = pop_reward
        reward += pop_reward

        # --- Time decay penalty (zone has been waiting) ---
        if zone.severity >= 0.5:
            decay = zone.time_since_incident * WEIGHTS["time_decay_penalty"]
            breakdown["time_decay_penalty"] = decay
            reward += decay

        # --- Priority alignment bonus ---
        correct_priority = _infer_correct_priority(zone)
        if action.priority == correct_priority:
            bonus = WEIGHTS["priority_alignment_bonus"]
            breakdown["priority_alignment_bonus"] = bonus
            reward += bonus

        # --- Hazard-resource affinity ---
        affine = RESOURCE_HAZARD_AFFINITY.get(ResourceType(action.resource_type), [])
        if HazardType(zone.hazard_type) in affine:
            bonus = 0.2 * action.quantity
            breakdown["hazard_affinity_bonus"] = bonus
            reward += bonus
        elif zone.hazard_type != HazardType.NONE:
            # Wrong resource type for the hazard
            breakdown["hazard_mismatch_penalty"] = -0.15
            reward -= 0.15

        # --- Over-deployment penalty ---
        already_deployed = sum(zone.resources_deployed.values())
        if already_deployed > 0 and action.quantity > already_deployed:
            pen = WEIGHTS["over_deploy_penalty"]
            breakdown["over_deploy_penalty"] = pen
            reward += pen

        # --- Reasoning bonus (encourages chain-of-thought) ---
        if action.reasoning and len(action.reasoning.strip()) > 10:
            bonus = WEIGHTS["reasoning_bonus"]
            breakdown["reasoning_bonus"] = bonus
            reward += bonus

        breakdown["total"] = round(reward, 4)
        return round(reward, 4), breakdown

    @staticmethod
    def terminal_reward(
        zones_resolved: int,
        total_zones: int,
        total_population_saved: int,
        steps_used: int,
        max_steps: int,
    ) -> float:
        """
        End-of-episode bonus/penalty added on top of step rewards.
        Encourages completing all zones and speed.
        """
        completion_ratio = zones_resolved / max(total_zones, 1)
        speed_bonus = max(0.0, (max_steps - steps_used) / max_steps) * 0.5

        terminal = (completion_ratio * 2.0) + speed_bonus
        return round(terminal, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_correct_priority(zone: ZoneState) -> str:
    """Map zone severity to the correct START triage label."""
    if zone.severity >= 0.8:
        return TriagePriority.IMMEDIATE.value
    elif zone.severity >= 0.5:
        return TriagePriority.URGENT.value
    elif zone.severity >= 0.2:
        return TriagePriority.DELAYED.value
    else:
        return TriagePriority.EXPECTANT.value
