# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
OpenEnv spec compliance tests — this is the most critical test file.

These tests mirror what `openenv validate` checks. All must pass before submission.
Run with: pytest tests/test_openenv_compliance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pydantic import BaseModel

from models import DisasterAction, DisasterObservation, DisasterState
from server.my_environment import DisasterTriageEnvironment
from tasks.registry import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return DisasterTriageEnvironment()


@pytest.fixture
def reset_env(env):
    obs = env.reset(seed=42, task_id="task_01_single_zone")
    return env, obs


# ---------------------------------------------------------------------------
# 1. Model compliance
# ---------------------------------------------------------------------------

class TestModels:
    def test_action_is_pydantic(self):
        assert issubclass(DisasterAction, BaseModel)

    def test_observation_is_pydantic(self):
        assert issubclass(DisasterObservation, BaseModel)

    def test_state_is_pydantic(self):
        assert issubclass(DisasterState, BaseModel)

    def test_action_has_required_fields(self):
        schema = DisasterAction.model_json_schema()
        assert "zone_id" in schema.get("properties", {})
        assert "resource_type" in schema.get("properties", {})

    def test_action_instantiates(self):
        a = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        assert a.zone_id == 0

    def test_observation_instantiates(self):
        env = DisasterTriageEnvironment()
        obs = env.reset(seed=0)
        assert isinstance(obs, DisasterObservation)
        assert obs.task_id != ""

    def test_action_json_roundtrip(self):
        a = DisasterAction(zone_id=1, resource_type="rescue_unit", quantity=2, priority="immediate")
        restored = DisasterAction.model_validate_json(a.model_dump_json())
        assert restored == a


# ---------------------------------------------------------------------------
# 2. Interface compliance: reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=42)
        assert isinstance(obs, DisasterObservation)

    def test_reset_with_seed_is_deterministic(self, env):
        obs1 = env.reset(seed=99)
        obs2 = env.reset(seed=99)
        assert obs1.model_dump() == obs2.model_dump()

    def test_reset_different_seeds_differ(self, env):
        obs1 = env.reset(seed=1)
        obs2 = env.reset(seed=2)
        # Extremely unlikely to be identical
        severities1 = [z.severity for z in obs1.zones]
        severities2 = [z.severity for z in obs2.zones]
        assert severities1 != severities2

    def test_reset_resets_step_count(self, env):
        env.reset(seed=42)
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        env.step(action)
        env.reset(seed=42)
        state = env.state()
        assert state.step_count == 0

    def test_reset_returns_valid_zones(self, env):
        obs = env.reset(seed=42)
        assert len(obs.zones) > 0
        for zone in obs.zones:
            assert 0.0 <= zone.severity <= 1.0
            assert zone.population_at_risk >= 0

    def test_reset_with_task_id(self, env):
        for task_id in TASK_REGISTRY:
            obs = env.reset(seed=42, task_id=task_id)
            assert obs.task_id == task_id


# ---------------------------------------------------------------------------
# 3. Interface compliance: step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_tuple(self, reset_env):
        env, obs = reset_env
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        result = env.step(action)
        assert len(result) == 4

    def test_step_observation_is_typed(self, reset_env):
        env, _ = reset_env
        action = DisasterAction(zone_id=0, resource_type="rescue_unit", quantity=1, priority="delayed")
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, DisasterObservation)

    def test_step_reward_is_float(self, reset_env):
        env, _ = reset_env
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        _, reward, _, _ = env.step(action)
        assert isinstance(reward, float)

    def test_step_done_is_bool(self, reset_env):
        env, _ = reset_env
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        _, _, done, _ = env.step(action)
        assert isinstance(done, bool)

    def test_step_info_is_dict(self, reset_env):
        env, _ = reset_env
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        _, _, _, info = env.step(action)
        assert isinstance(info, dict)

    def test_step_increments_step_count(self, reset_env):
        env, _ = reset_env
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        env.step(action)
        assert env.state().step_count == 1

    def test_step_invalid_zone_penalizes(self, reset_env):
        env, obs = reset_env
        bad_zone = len(obs.zones) + 99
        action = DisasterAction(zone_id=bad_zone, resource_type="medical_team", quantity=1, priority="urgent")
        _, reward, _, info = env.step(action)
        assert reward < 0
        assert "error" in info


# ---------------------------------------------------------------------------
# 4. Interface compliance: state()
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_typed_object(self, reset_env):
        env, _ = reset_env
        state = env.state()
        assert isinstance(state, DisasterState)

    def test_state_has_episode_id(self, reset_env):
        env, _ = reset_env
        state = env.state()
        assert state.episode_id != ""

    def test_state_step_count_matches(self, reset_env):
        env, _ = reset_env
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        env.step(action)
        env.step(action)
        state = env.state()
        assert state.step_count == 2


# ---------------------------------------------------------------------------
# 5. Task grader compliance
# ---------------------------------------------------------------------------

class TestGraders:
    @pytest.mark.parametrize("task_id", list(TASK_REGISTRY.keys()))
    def test_grader_returns_score_in_range(self, env, task_id):
        obs = env.reset(seed=42, task_id=task_id)
        # Run a few steps
        for _ in range(5):
            if obs.done:
                break
            zone_id = obs.zones[0].zone_id if obs.zones else 0
            action = DisasterAction(zone_id=zone_id, resource_type="medical_team", quantity=1, priority="urgent")
            obs, _, done, _ = env.step(action)

        result = env.grade()
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range for {task_id}"

    @pytest.mark.parametrize("task_id", list(TASK_REGISTRY.keys()))
    def test_grader_is_deterministic(self, task_id):
        """Same seed + same actions → same score. Always."""
        def run_episode(seed):
            e = DisasterTriageEnvironment()
            obs = e.reset(seed=seed, task_id=task_id)
            for _ in range(5):
                if obs.done:
                    break
                zone_id = obs.zones[0].zone_id if obs.zones else 0
                action = DisasterAction(
                    zone_id=zone_id, resource_type="medical_team",
                    quantity=1, priority="urgent"
                )
                obs, _, _, _ = e.step(action)
            return e.grade().score

        score1 = run_episode(seed=7)
        score2 = run_episode(seed=7)
        assert score1 == score2, f"Non-deterministic grader for {task_id}"

    @pytest.mark.parametrize("task_id", list(TASK_REGISTRY.keys()))
    def test_grader_has_breakdown(self, env, task_id):
        env.reset(seed=42, task_id=task_id)
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        env.step(action)
        result = env.grade()
        assert isinstance(result.breakdown, dict)
        assert len(result.breakdown) > 0

    def test_empty_episode_scores_minimum(self, env):
    env.reset(seed=42, task_id="task_01_single_zone")
        result = env.grade()
        assert result.score == 0.01  
# ---------------------------------------------------------------------------
# 6. Episode lifecycle
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_episode_terminates_at_max_steps(self, env):
        obs = env.reset(seed=42, task_id="task_01_single_zone")
        max_steps = obs.max_steps
        action = DisasterAction(zone_id=0, resource_type="supply_drop", quantity=1, priority="delayed")
        done = False
        steps = 0
        while not done and steps < max_steps + 5:
            obs, _, done, _ = env.step(action)
            steps += 1
        assert done, "Episode never terminated"
        assert steps <= max_steps + 1

    def test_cumulative_reward_accumulates(self, env):
        obs = env.reset(seed=42, task_id="task_01_single_zone")
        action = DisasterAction(zone_id=0, resource_type="medical_team", quantity=1, priority="urgent")
        env.step(action)
        obs2, _, _, _ = env.step(action)
        assert obs2.cumulative_reward != 0.0

    def test_three_tasks_exist(self):
        assert len(TASK_REGISTRY) >= 3

    def test_task_difficulties_span_easy_to_hard(self):
        difficulties = {cls().difficulty for cls in TASK_REGISTRY.values()}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties
