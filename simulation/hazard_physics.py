# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Hazard physics — pure, stateless functions.

No side effects. All state mutation happens in disaster_env.py.
These functions are deterministic given the same inputs.
"""

from __future__ import annotations

import numpy as np
from models import ZoneState, HazardEvent, HazardType, ResourceType


def apply_time_decay(zones: list[ZoneState], step: int) -> list[ZoneState]:
    """
    Increase time_since_incident for unresolved zones each step.
    Slightly escalate severity for zones with active hazards left unattended.
    """
    updated = []
    for z in zones:
        if z.resolved:
            updated.append(z)
            continue

        new_time = z.time_since_incident + 1

        # Hazard escalation: flood/fire severity grows slowly without intervention
        severity_delta = 0.0
        if z.hazard_type in (HazardType.FLOOD, HazardType.FIRE) and new_time > 3:
            severity_delta = 0.02  # +2% per neglected step

        new_severity = min(1.0, z.severity + severity_delta)

        updated.append(ZoneState(
            **{**z.model_dump(), "time_since_incident": new_time, "severity": new_severity}
        ))
    return updated


def apply_cascade_spread(
    zones: list[ZoneState],
    hazards: list[HazardEvent],
    current_step: int,
    rng: np.random.Generator,
) -> tuple[list[ZoneState], list[HazardEvent]]:
    """
    For each active hazard, roll to spread to adjacent zones.
    Mutates a copy of zones and returns updated zones + hazard list.
    """
    zone_map = {z.zone_id: z for z in zones}
    new_hazards: list[HazardEvent] = []

    for hazard in hazards:
        if current_step < hazard.step_triggered:
            new_hazards.append(hazard)
            continue

        newly_affected: list[int] = []
        for target_id in hazard.affected_zones:
            if target_id not in zone_map:
                continue
            target = zone_map[target_id]
            if target.resolved:
                continue
            if rng.random() < hazard.spread_probability:
                # Spread: escalate severity of target zone
                new_sev = min(1.0, target.severity + 0.15)
                zone_map[target_id] = ZoneState(
                    **{**target.model_dump(),
                       "severity": new_sev,
                       "hazard_type": hazard.hazard_type}
                )
                newly_affected.append(target_id)

        updated_hazard = HazardEvent(
            **{**hazard.model_dump(),
               "affected_zones": [z for z in hazard.affected_zones if z not in newly_affected]}
        )
        if updated_hazard.affected_zones:
            new_hazards.append(updated_hazard)

    return list(zone_map.values()), new_hazards


def resolve_zone(zone: ZoneState, action_resource: ResourceType, quantity: int) -> ZoneState:
    """
    Apply a resource deployment to a zone, potentially resolving it.

    Resolution logic:
      - A zone is resolved when severity-weighted resource coverage >= threshold.
      - Threshold = ceil(severity * 3) units of the correct resource type.
    """
    affinity_map = {
        ResourceType.MEDICAL_TEAM:  [HazardType.EARTHQUAKE, HazardType.FIRE, HazardType.NONE],
        ResourceType.RESCUE_UNIT:   [HazardType.FLOOD, HazardType.EARTHQUAKE, HazardType.NONE],
        ResourceType.SUPPLY_DROP:   [HazardType.FLOOD, HazardType.NONE],
        ResourceType.HAZMAT_CREW:   [HazardType.HAZMAT, HazardType.FIRE],
    }

    new_deployed = dict(zone.resources_deployed)
    new_deployed[action_resource.value] = new_deployed.get(action_resource.value, 0) + quantity

    # Compute resolution threshold
    threshold = max(1, int(zone.severity * 3))
    total_affine_units = sum(
        count for rtype, count in new_deployed.items()
        if HazardType(zone.hazard_type) in affinity_map.get(ResourceType(rtype), [])
    )

    resolved = total_affine_units >= threshold

    # If resolved, severity drops to 0
    new_severity = 0.0 if resolved else zone.severity

    return ZoneState(
        **{**zone.model_dump(),
           "resources_deployed": new_deployed,
           "resolved": resolved,
           "severity": new_severity}
    )
