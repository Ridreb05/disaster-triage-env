# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
DisasterTriageEnvironment — the core environment class.

Extends openenv.core.env_server.interfaces.Environment.
This is the file that wires the task system, simulation engine,
and reward calculator into the OpenEnv step/reset/state contract.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    DisasterAction, DisasterObservation, DisasterState,
    ResourceInventory, ZoneState, HazardEvent, ResourceType
)
from simulation.incident_generator import IncidentGenerator
from simulation.hazard_physics import apply_time_decay, apply_cascade_spread, resolve_zone
from simulation.reward_calculator import RewardCalculator
from tasks.registry import get_task, list_tasks
from tasks.base_task import GraderResult


class DisasterTriageEnvironment(Environment):
    """
    A disaster response triage environment where an agent allocates
    emergency resources across incident zones under time pressure
    and cascading hazard conditions.

    Implements the full OpenEnv interface:
        reset(seed, task_id) → DisasterObservation
        step(action)         → (DisasterObservation, reward, done, info)
        state()              → DisasterState
    """

    def __init__(self):
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._task_id: str = "task_01_single_zone"
        self._seed: int = 0
        self._max_steps: int = 20

        # Episode state
        self._zones: list[ZoneState] = []
        self._resources: ResourceInventory = ResourceInventory()
        self._hazards: list[HazardEvent] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False

        # Episode history for grading
        self._episode_history: list[dict] = []

        # RNG (re-seeded on reset)
        import numpy as np
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> DisasterObservation:
        """
        Start a new episode.

        Args:
            seed:    RNG seed for deterministic scenario generation.
                     If None, uses a random seed.
            task_id: One of the registered task IDs. Defaults to task_01.
        """
        import numpy as np

        self._seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        self._task_id = task_id or "task_01_single_zone"
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_history = []

        task = get_task(self._task_id)
        self._max_steps = task.scenario_config.max_steps

        generator = IncidentGenerator(seed=self._seed, config=task.scenario_config)
        self._zones, self._resources, self._hazards = generator.generate()
        self._rng = np.random.default_rng(self._seed + 1)  # separate RNG for physics

        return self._build_observation()

    def step(self, action: DisasterAction) -> tuple[DisasterObservation, float, bool, dict]:
        """
        Execute one action in the environment.

        Returns:
            observation — updated world state
            reward      — per-step reward (rich partial signal)
            done        — True when episode ends
            info        — auxiliary info including reward breakdown
        """
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already done. Call reset()."}

        # Validate action
        info: dict = {}
        zone_id = action.zone_id
        if zone_id < 0 or zone_id >= len(self._zones):
            reward = -0.5
            info["error"] = f"Invalid zone_id {zone_id}. Valid range: 0–{len(self._zones)-1}"
            self._step_count += 1
            obs = self._build_observation()
            self._record_step(action, obs, reward)
            return obs, reward, self._check_done(), info

        target_zone = self._zones[zone_id]

        # Consume resources (silently cap at available)
        resource_type = ResourceType(action.resource_type)
        available = self._resources.available(resource_type)
        actual_quantity = min(action.quantity, available)
        if actual_quantity <= 0:
            reward = -0.1
            info["warning"] = f"No {action.resource_type} remaining."
            self._step_count += 1
            obs = self._build_observation()
            self._record_step(action, obs, reward)
            return obs, reward, self._check_done(), info

        # Compute reward BEFORE resolving (uses pre-action zone state)
        reward, reward_breakdown = RewardCalculator.compute(
            action=action,
            zone=target_zone,
            resources_before=self._resources,
            elapsed_steps=self._step_count,
        )

        # Apply resource consumption
        self._resources = self._resources.consume(resource_type, actual_quantity)

        # Apply resolution logic
        self._zones[zone_id] = resolve_zone(target_zone, resource_type, actual_quantity)

        # Apply time decay and hazard cascades
        self._zones = apply_time_decay(self._zones, self._step_count)
        self._zones, self._hazards = apply_cascade_spread(
            self._zones, self._hazards, self._step_count, self._rng
        )

        self._step_count += 1
        self._cumulative_reward += reward

        done = self._check_done()

        # Add terminal reward if episode ending
        if done:
            terminal = RewardCalculator.terminal_reward(
                zones_resolved=sum(1 for z in self._zones if z.resolved),
                total_zones=len(self._zones),
                total_population_saved=sum(
                    z.population_at_risk for z in self._zones if z.resolved
                ),
                steps_used=self._step_count,
                max_steps=self._max_steps,
            )
            reward += terminal
            self._cumulative_reward += terminal
            info["terminal_reward"] = terminal

        self._done = done
        info["reward_breakdown"] = reward_breakdown
        info["actual_quantity"] = actual_quantity

        obs = self._build_observation()
        self._record_step(action, obs, reward)
        return obs, round(reward, 4), done, info

    def state(self) -> DisasterState:
        """Returns current episode metadata."""
        return DisasterState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            scenario_seed=self._seed,
            total_population_at_risk=sum(z.population_at_risk for z in self._zones),
            zones_resolved=sum(1 for z in self._zones if z.resolved),
            total_zones=len(self._zones),
            resources_deployed_total=sum(
                sum(r.values()) for z in self._zones
                for r in [z.resources_deployed]
            ),
            current_score=self._get_current_score(),
            hazard_events_triggered=len(self._hazards),
        )

    # ------------------------------------------------------------------
    # Additional endpoints (called by app.py)
    # ------------------------------------------------------------------

    def grade(self) -> GraderResult:
        """Grade the current completed episode."""
        task = get_task(self._task_id)
        return task.grade(self._episode_history)

    def list_tasks(self) -> list[dict]:
        return list_tasks()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> DisasterObservation:
        return DisasterObservation(
            zones=self._zones,
            resources_remaining=self._resources,
            elapsed_steps=self._step_count,
            max_steps=self._max_steps,
            active_hazards=self._hazards,
            cumulative_reward=round(self._cumulative_reward, 4),
            task_id=self._task_id,
            scenario_seed=self._seed,
            done=self._done,
            info={},
        )

    def _check_done(self) -> bool:
        all_resolved = all(z.resolved for z in self._zones)
        out_of_steps = self._step_count >= self._max_steps
        out_of_resources = all(
            self._resources.available(rt) == 0
            for rt in ResourceType
        )
        return all_resolved or out_of_steps or out_of_resources

    def _record_step(self, action: DisasterAction, obs: DisasterObservation, reward: float):
        self._episode_history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "observation": obs.model_dump(),
            "reward": reward,
            "done": self._done,
        })

    def _get_current_score(self) -> float:
        if not self._episode_history:
            return 0.0
        task = get_task(self._task_id)
        try:
            result = task.grade(self._episode_history)
            return result.score
        except Exception:
            return 0.0
