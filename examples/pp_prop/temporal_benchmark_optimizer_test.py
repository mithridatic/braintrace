"""Tests for optimizer schedule, clipping, and search grids."""

import math

import numpy as np

from temporal_benchmark_optimizer import (
    SealedLearningRateSchedule,
    clip_leaves,
    scheduled_learning_rate,
    successive_halving_grid,
)


def test_schedule_has_warmup_hold_and_fifteen_percent_floor() -> None:
    rates = [scheduled_learning_rate(index, 100, 1.0) for index in range(100)]

    assert rates[0] == 0.1
    assert math.isclose(rates[9], 1.0)
    assert all(math.isclose(rate, 1.0) for rate in rates[9:71])
    assert math.isclose(rates[-1], 0.15)


def test_jit_schedule_matches_scalar_schedule_at_every_update() -> None:
    schedule = SealedLearningRateSchedule(1.0, 100)
    actual = []
    for _ in range(100):
        actual.append(float(schedule.current_lrs.value[0]))
        schedule.step()

    expected = [scheduled_learning_rate(index, 100, 1.0) for index in range(100)]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_group_clipping_reports_event_and_exact_norm() -> None:
    clipped, raw, final, event = clip_leaves([np.asarray([3.0, 4.0])], 1.0)

    assert raw == 5.0
    assert math.isclose(final, 1.0)
    assert event
    np.testing.assert_allclose(clipped[0], [0.6, 0.8])


def test_successive_halving_grid_has_all_twenty_seven_combinations() -> None:
    grid = successive_halving_grid()

    assert len(grid) == 27
    assert len(set(grid)) == 27
