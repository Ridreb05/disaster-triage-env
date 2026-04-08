# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Per-task grader validation.

Verifies each task grader against known action sequences to prevent
regressions in grading logic.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from server.my_environment import DisasterTriageEnvironment
from models import DisasterAction
from tasks.registry import TASK_REGISTRY, get_task


def _run_optimal_episode(task_id: str, seed: int = 42) -> float:
    """
    Run a near-optimal episode: always target the highest-severity
    accessible zone with the most resources available.
    Returns grader score.
    """
    env = DisasterTriageEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    while not obs.done:
        candidates = [
            z for z in obs.zones
            if z.is_accessible and not z.resolved
        ]
        if not candidates:
            break
        target = max(candidates, key=lambda z: z.severity)

        hazard_resource_map = {
            "earthquake": "medical_team",
            "flood":      "rescue_unit",
            "hazmat":     "hazmat_crew",
            "fire":       "medical_team",
            "none":       "medical_team",
        }
        resource = hazard_resource_map.get(target.hazard_type, "medical_team")

        # Fall back if resource exhausted
        resources = obs.resources_remaining.model_dump()
        if resources.get(resource, 0) == 0:
            resource = next(
                (k for k, v in resources.items() if isinstance(v, int) and v > 0),
                "supply_drop"
            )

        severity = target.severity
        priority = (
            "immediate" if severity >= 0.8
            else "urgent" if severity >= 0.5
            else "delayed"
        )

        action = DisasterAction(
            zone_id=target.zone_id,
            resource_type=resource,
            quantity=min(2, max(1, resources.get(resource, 1))),
            priority=priority,
            reasoning=f"Targeting zone {target.zone_id} with severity {severity:.2f}.",
        )
        obs, _, _, _ = env.step(action)

    return env.grade().score


def _run_random_episode(task_id: str, seed: int = 0) -> float:
    """Run a random (bad) episode to verify grader discriminates."""
    env = DisasterTriageEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    for _ in range(3):
        if obs.done:
            break
        # Deliberately bad: always deploy to zone 0 with wrong resource
        action = DisasterAction(
            zone_id=0, resource_type="supply_drop", quantity=1, priority="delayed"
        )
        obs, _, _, _ = env.step(action)

    return env.grade().score


class TestTask01:
    def test_optimal_beats_random(self):
        optimal = _run_optimal_episode("task_01_single_zone", seed=42)
        random = _run_random_episode("task_01_single_zone", seed=42)
        assert optimal >= random, "Optimal agent should score >= random agent"

    def test_optimal_score_is_reasonable(self):
        score = _run_optimal_episode("task_01_single_zone", seed=42)
        assert score >= 0.3, f"Optimal agent scored only {score:.4f} on easy task"

    def test_no_actions_scores_zero(self):
        env = DisasterTriageEnvironment()
        env.reset(seed=42, task_id="task_01_single_zone")
        assert env.grade().score == 0.0

    def test_task_config_is_easy(self):
        task = get_task("task_01_single_zone")
        assert task.scenario_config.num_zones <= 5
        assert task.scenario_config.max_steps <= 10
        assert task.difficulty == "easy"


class TestTask02:
    def test_optimal_beats_random(self):
        optimal = _run_optimal_episode("task_02_multi_zone", seed=42)
        random = _run_random_episode("task_02_multi_zone", seed=42)
        assert optimal >= random

    def test_task_has_5_zones(self):
        task = get_task("task_02_multi_zone")
        assert task.scenario_config.num_zones == 5
        assert task.difficulty == "medium"

    def test_grader_penalizes_low_resolution(self):
        """Only resolving 1 zone should score below passing."""
        env = DisasterTriageEnvironment()
        obs = env.reset(seed=42, task_id="task_02_multi_zone")
        # Deliberately only act once
        action = DisasterAction(
            zone_id=obs.zones[0].zone_id,
            resource_type="medical_team",
            quantity=1,
            priority="urgent"
        )
        env.step(action)
        score = env.grade().score
        assert score < 0.7, f"Expected lower score for minimal effort, got {score}"


class TestTask03:
    def test_hard_task_has_10_zones(self):
        task = get_task("task_03_cascading_hazards")
        assert task.scenario_config.num_zones == 10
        assert task.difficulty == "hard"

    def test_hard_task_has_hazard_probability(self):
        task = get_task("task_03_cascading_hazards")
        assert task.scenario_config.hazard_probability > 0

    def test_optimal_scores_higher_than_random(self):
        optimal = _run_optimal_episode("task_03_cascading_hazards", seed=42)
        random = _run_random_episode("task_03_cascading_hazards", seed=42)
        assert optimal >= random


class TestRegistry:
    def test_all_tasks_have_unique_ids(self):
        ids = list(TASK_REGISTRY.keys())
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_action_schema(self):
        for task_id, cls in TASK_REGISTRY.items():
            task = cls()
            schema = task.action_schema()
            assert "properties" in schema, f"{task_id} missing action schema properties"

    def test_list_tasks_returns_all(self):
        from tasks.registry import list_tasks
        tasks = list_tasks()
        assert len(tasks) == len(TASK_REGISTRY)
        for t in tasks:
            assert "task_id" in t
            assert "difficulty" in t
            assert "action_schema" in t
