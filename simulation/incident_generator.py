# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Seeded incident generator — the heart of deterministic reproducibility.

All randomness flows through a single numpy.random.Generator seeded at
episode start. The same seed always produces the identical scenario,
satisfying the "baseline reproduces" requirement from the hackathon spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from models import (
    ZoneState, ResourceInventory, HazardEvent,
    HazardType, ResourceType
)


# ---------------------------------------------------------------------------
# Scenario configuration dataclass (injected per task)
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Describes the parameters for a generated scenario.
    Passed in by each task to control difficulty.
    """
    num_zones: int = 5
    max_steps: int = 20
    base_resources: dict = field(default_factory=lambda: {
        "medical_team": 4,
        "rescue_unit": 3,
        "supply_drop": 5,
        "hazmat_crew": 2,
    })
    hazard_probability: float = 0.0        # Chance any zone spawns a cascading hazard
    cascade_spread_probability: float = 0.0
    min_severity: float = 0.1
    max_severity: float = 1.0
    blocked_zone_probability: float = 0.0  # Chance a zone starts inaccessible
    population_range: tuple[int, int] = (50, 500)


# ---------------------------------------------------------------------------
# Pre-baked named scenarios (used by the Gradio demo / fixtures)
# ---------------------------------------------------------------------------

NAMED_SCENARIOS: dict[str, ScenarioConfig] = {
    "cyclone_amphan": ScenarioConfig(
        num_zones=8,
        max_steps=25,
        base_resources={
            "medical_team": 6,
            "rescue_unit": 5,
            "supply_drop": 10,
            "hazmat_crew": 1,
        },
        hazard_probability=0.4,
        cascade_spread_probability=0.3,
        blocked_zone_probability=0.25,
        min_severity=0.3,
        max_severity=1.0,
        population_range=(200, 2000),
    ),
    "earthquake_nepal": ScenarioConfig(
        num_zones=10,
        max_steps=30,
        base_resources={
            "medical_team": 5,
            "rescue_unit": 8,
            "supply_drop": 6,
            "hazmat_crew": 3,
        },
        hazard_probability=0.5,
        cascade_spread_probability=0.2,
        blocked_zone_probability=0.4,
        min_severity=0.4,
        max_severity=1.0,
        population_range=(100, 3000),
    ),
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class IncidentGenerator:
    """
    Generates a fully-described disaster scenario from a seed and config.

    Usage:
        gen = IncidentGenerator(seed=42, config=ScenarioConfig(num_zones=5))
        zones, resources, hazards = gen.generate()
    """

    def __init__(self, seed: int, config: Optional[ScenarioConfig] = None):
        self.seed = seed
        self.config = config or ScenarioConfig()
        self._rng = np.random.default_rng(seed)

    def generate(self) -> tuple[list[ZoneState], ResourceInventory, list[HazardEvent]]:
        """
        Returns:
            zones        — initial per-zone world state
            resources    — starting resource inventory
            hazards      — list of pre-seeded hazard events (may be empty)
        """
        zones = self._generate_zones()
        resources = self._generate_resources()
        hazards = self._generate_hazards(zones)
        return zones, resources, hazards

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_zones(self) -> list[ZoneState]:
        zones = []
        cfg = self.config

        # Severity weights: bias toward critical incidents for more interesting episodes
        raw_severities = self._rng.beta(a=2.0, b=1.5, size=cfg.num_zones)
        severities = np.clip(
            raw_severities * (cfg.max_severity - cfg.min_severity) + cfg.min_severity,
            cfg.min_severity,
            cfg.max_severity,
        )

        hazard_pool = [HazardType.FLOOD, HazardType.EARTHQUAKE,
                       HazardType.HAZMAT, HazardType.FIRE, HazardType.NONE]
        hazard_weights = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
        # Use integer index selection — rng.choice on Enum objects produces
        # truncated numpy.str_ values which fail Pydantic validation.
        hazard_indices = self._rng.choice(len(hazard_pool), size=cfg.num_zones, p=hazard_weights)

        for i in range(cfg.num_zones):
            hazard = hazard_pool[int(hazard_indices[i])]
            pop = int(self._rng.integers(*cfg.population_range))
            blocked = bool(self._rng.random() < cfg.blocked_zone_probability)

            zones.append(ZoneState(
                zone_id=i,
                severity=float(round(severities[i], 3)),
                population_at_risk=pop,
                hazard_type=hazard,
                is_accessible=not blocked,
                resources_deployed={},
                time_since_incident=int(self._rng.integers(0, 5)),
                resolved=False,
            ))

        return zones

    def _generate_resources(self) -> ResourceInventory:
        base = self.config.base_resources
        # Slight jitter so resources are not perfectly balanced
        jitter = lambda v: max(1, v + int(self._rng.integers(-1, 2)))
        return ResourceInventory(
            medical_team=jitter(base.get("medical_team", 4)),
            rescue_unit=jitter(base.get("rescue_unit", 3)),
            supply_drop=jitter(base.get("supply_drop", 5)),
            hazmat_crew=jitter(base.get("hazmat_crew", 2)),
        )

    def _generate_hazards(self, zones: list[ZoneState]) -> list[HazardEvent]:
        hazards: list[HazardEvent] = []
        cfg = self.config

        if cfg.hazard_probability <= 0.0:
            return hazards

        for zone in zones:
            if self._rng.random() < cfg.hazard_probability:
                # Determine which adjacent zones could be affected
                adjacent = [
                    z.zone_id for z in zones
                    if abs(z.zone_id - zone.zone_id) == 1
                ]
                hazards.append(HazardEvent(
                    hazard_type=zone.hazard_type if zone.hazard_type != HazardType.NONE
                               else HazardType.FLOOD,
                    origin_zone=zone.zone_id,
                    affected_zones=adjacent,
                    spread_probability=float(self._rng.uniform(
                        0.0, cfg.cascade_spread_probability
                    )),
                    step_triggered=int(self._rng.integers(2, max(3, cfg.max_steps // 3))),
                ))

        return hazards
