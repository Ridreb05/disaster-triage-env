# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Typed models for the Disaster Triage Environment.

Extends openenv.core.env_server.types with domain-specific fields.
All models are Pydantic v2 for full JSON schema generation and validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResourceType(str, Enum):
    MEDICAL_TEAM = "medical_team"
    RESCUE_UNIT = "rescue_unit"
    SUPPLY_DROP = "supply_drop"
    HAZMAT_CREW = "hazmat_crew"


class TriagePriority(str, Enum):
    """START triage classification."""
    IMMEDIATE = "immediate"    # Life-threatening, saveable — red tag
    URGENT = "urgent"          # Serious but stable — yellow tag
    DELAYED = "delayed"        # Minor injuries — green tag
    EXPECTANT = "expectant"    # Unsurvivable without major resources — black tag


class HazardType(str, Enum):
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    HAZMAT = "hazmat"
    FIRE = "fire"
    NONE = "none"


# ---------------------------------------------------------------------------
# Sub-models (used inside Observation)
# ---------------------------------------------------------------------------

class ZoneState(BaseModel):
    """Snapshot of one disaster zone."""
    zone_id: int = Field(..., description="Zone index 0-N")
    severity: float = Field(..., ge=0.0, le=1.0, description="Incident severity 0–1")
    population_at_risk: int = Field(..., ge=0, description="Estimated affected population")
    hazard_type: HazardType = Field(default=HazardType.NONE)
    is_accessible: bool = Field(default=True, description="False if road/access is blocked")
    resources_deployed: dict[str, int] = Field(
        default_factory=dict,
        description="resource_type → units already deployed here"
    )
    time_since_incident: int = Field(default=0, description="Steps since incident was reported")
    resolved: bool = Field(default=False, description="True when zone is fully triaged")

    class Config:
        use_enum_values = True


class ResourceInventory(BaseModel):
    """Available resources remaining in the agent's pool."""
    medical_team: int = Field(default=0, ge=0)
    rescue_unit: int = Field(default=0, ge=0)
    supply_drop: int = Field(default=0, ge=0)
    hazmat_crew: int = Field(default=0, ge=0)

    def available(self, resource_type: ResourceType) -> int:
        return getattr(self, resource_type.value, 0)

    def consume(self, resource_type: ResourceType, quantity: int) -> "ResourceInventory":
        data = self.model_dump()
        data[resource_type.value] = max(0, data[resource_type.value] - quantity)
        return ResourceInventory(**data)


class HazardEvent(BaseModel):
    """A cascading hazard event spreading between zones."""
    hazard_type: HazardType
    origin_zone: int
    affected_zones: list[int] = Field(default_factory=list)
    spread_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    step_triggered: int = Field(default=0)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Core OpenEnv types
# ---------------------------------------------------------------------------

class DisasterAction(Action):
    """
    Agent action: deploy a resource to a specific zone with a triage priority.

    Example:
        DisasterAction(
            zone_id=2,
            resource_type="medical_team",
            quantity=2,
            priority="immediate"
        )
    """
    zone_id: int = Field(..., ge=0, description="Target zone index")
    resource_type: ResourceType = Field(..., description="Type of resource to deploy")
    quantity: int = Field(default=1, ge=1, le=5, description="Units to deploy (1–5)")
    priority: TriagePriority = Field(
        default=TriagePriority.URGENT,
        description="START triage priority classification"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought justification (used for scoring bonus)"
    )

    class Config:
        use_enum_values = True


class DisasterObservation(Observation):
    """
    Full world state observation returned after every reset/step.

    Designed to be directly consumable by an LLM agent as structured JSON.
    """
    zones: list[ZoneState] = Field(..., description="Per-zone state array")
    resources_remaining: ResourceInventory = Field(...)
    elapsed_steps: int = Field(default=0, ge=0)
    max_steps: int = Field(..., description="Episode step budget")
    active_hazards: list[HazardEvent] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    task_id: str = Field(..., description="Active task identifier")
    scenario_seed: int = Field(..., description="Seed used for reproducibility")
    done: bool = Field(default=False)
    info: dict = Field(default_factory=dict, description="Auxiliary info dict")

    @property
    def critical_zones(self) -> list[ZoneState]:
        """Zones with severity >= 0.7 that are unresolved."""
        return [z for z in self.zones if z.severity >= 0.7 and not z.resolved]

    @property
    def accessible_zones(self) -> list[ZoneState]:
        return [z for z in self.zones if z.is_accessible and not z.resolved]

    class Config:
        use_enum_values = True


class DisasterState(State):
    """
    Internal episode state returned by state() endpoint.
    Extends the core State (episode_id, step_count).
    """
    task_id: str = Field(default="")
    scenario_seed: int = Field(default=0)
    total_population_at_risk: int = Field(default=0)
    zones_resolved: int = Field(default=0)
    total_zones: int = Field(default=0)
    resources_deployed_total: int = Field(default=0)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hazard_events_triggered: int = Field(default=0)