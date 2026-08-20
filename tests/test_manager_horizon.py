"""Tests for manager-horizon / budget planning (HRL cycle overshoot guard)."""

from __future__ import annotations

import pytest

from agents import plan_manager_horizon


def test_plan_shrinks_n_steps_to_fit_min_manager_updates():
    # Old Stage 0 failure mode: 256 * 8 * 2052 = 4.2M > 250k budget.
    eff, cycle, expected = plan_manager_horizon(
        250_000, n_steps=256, goal_every=8, n_envs=2052, min_manager_updates=10,
    )
    assert eff == 1  # only 1 manager window/cycle fits 10x into 250k
    assert cycle == 8 * 2052
    assert expected >= 10
    assert expected * cycle <= 250_000


def test_plan_keeps_n_steps_when_budget_ample():
    eff, cycle, expected = plan_manager_horizon(
        2_000_000, n_steps=16, goal_every=8, n_envs=1026, min_manager_updates=15,
    )
    assert eff == 16
    assert cycle == 16 * 8 * 1026
    assert expected >= 15


def test_plan_rejects_impossible_budget():
    with pytest.raises(ValueError, match="too small"):
        plan_manager_horizon(
            10_000, n_steps=16, goal_every=8, n_envs=2052, min_manager_updates=10,
        )
