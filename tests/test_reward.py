# Copyright (c) 2024 Debanik Das. BSD-3-Clause License.
"""
Reward function tests.

Verifies: partial signal, monotonicity, penalty correctness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import DisasterAction, ZoneState, ResourceInventory, HazardType
from simulation.reward_calculator import RewardCalculator


def _make_zone(severity=0.8, hazard="earthquake", accessible=True, resolved=False, time=0):
    return ZoneState(
        zone_id=0,
        severity=severity,
        population_at_risk=500,
        hazard_type=hazard,
        is_accessible=accessible,
        resolved=resolved,
        time_since_incident=time,
    )


def _make_action(zone_id=0, resource="medical_team", qty=2, priority="immediate", reasoning=None):
    return DisasterAction(
        zone_id=zone_id, resource_type=resource,
        quantity=qty, priority=priority, reasoning=reasoning
    )


class TestRewardSignal:
    def test_reward_is_float(self):
        zone = _make_zone()
        action = _make_action()
        reward, _ = RewardCalculator.compute(action, zone, ResourceInventory(), 0)
        assert isinstance(reward, float)

    def test_inaccessible_zone_penalized(self):
        zone = _make_zone(accessible=False)
        action = _make_action()
        reward, breakdown = RewardCalculator.compute(action, zone, ResourceInventory(), 0)
        assert reward < 0
        assert "inaccessible_penalty" in breakdown

    def test_resolved_zone_penalized(self):
        zone = _make_zone(resolved=True)
        action = _make_action()
        reward, breakdown = RewardCalculator.compute(action, zone, ResourceInventory(), 0)
        assert reward < 0
        assert "resource_waste_penalty" in breakdown

    def test_high_severity_earns_more_than_low(self):
        high_zone = _make_zone(severity=0.9)
        low_zone = _make_zone(severity=0.1)
        action = _make_action()
        r_high, _ = RewardCalculator.compute(action, high_zone, ResourceInventory(), 0)
        r_low, _ = RewardCalculator.compute(action, low_zone, ResourceInventory(), 0)
        assert r_high > r_low

    def test_priority_alignment_bonus_awarded(self):
        zone = _make_zone(severity=0.85)  # should be IMMEDIATE
        action_correct = _make_action(priority="immediate")
        action_wrong = _make_action(priority="delayed")
        r_correct, bd_correct = RewardCalculator.compute(action_correct, zone, ResourceInventory(), 0)
        r_wrong, _ = RewardCalculator.compute(action_wrong, zone, ResourceInventory(), 0)
        assert r_correct > r_wrong
        assert "priority_alignment_bonus" in bd_correct

    def test_hazard_affinity_bonus(self):
        zone = _make_zone(hazard="flood")
        good_action = _make_action(resource="rescue_unit")
        bad_action = _make_action(resource="hazmat_crew")
        r_good, _ = RewardCalculator.compute(good_action, zone, ResourceInventory(), 0)
        r_bad, _ = RewardCalculator.compute(bad_action, zone, ResourceInventory(), 0)
        assert r_good > r_bad

    def test_time_decay_increases_penalty(self):
        zone_fresh = _make_zone(severity=0.7, time=0)
        zone_stale = _make_zone(severity=0.7, time=10)
        action = _make_action()
        r_fresh, _ = RewardCalculator.compute(action, zone_fresh, ResourceInventory(), 0)
        r_stale, _ = RewardCalculator.compute(action, zone_stale, ResourceInventory(), 0)
        assert r_fresh > r_stale

    def test_reasoning_bonus_awarded(self):
        zone = _make_zone()
        action_with = _make_action(reasoning="Targeting highest severity zone for medical response.")
        action_without = _make_action(reasoning=None)
        r_with, _ = RewardCalculator.compute(action_with, zone, ResourceInventory(), 0)
        r_without, _ = RewardCalculator.compute(action_without, zone, ResourceInventory(), 0)
        assert r_with > r_without

    def test_breakdown_has_total(self):
        zone = _make_zone()
        action = _make_action()
        _, breakdown = RewardCalculator.compute(action, zone, ResourceInventory(), 0)
        assert "total" in breakdown
        assert isinstance(breakdown["total"], float)

    def test_terminal_reward_positive_for_full_completion(self):
        t = RewardCalculator.terminal_reward(
            zones_resolved=5, total_zones=5,
            total_population_saved=1000,
            steps_used=10, max_steps=20,
        )
        assert t > 0

    def test_terminal_reward_penalizes_no_completion(self):
        t = RewardCalculator.terminal_reward(
            zones_resolved=0, total_zones=5,
            total_population_saved=0,
            steps_used=20, max_steps=20,
        )
        assert t == 0.0 or t < 1.0
