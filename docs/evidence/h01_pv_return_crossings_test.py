"""Check boundary, repeated, and missing downward crossing observations."""

import numpy as np
import pytest

from docs.evidence.h01_pv_return_crossings import downward_crossings


def test_nonuniform_crossings_use_crossing_time_for_interval_membership():
    """Retain a crossing whose bracketing sample precedes the interval start."""
    time = np.array([0., 2., 3., 7.])
    voltage = np.array([10., -10., 10., -10.])
    row, missing = downward_crossings(time, voltage, .5, 6., (0., -20.))
    assert row["first_time_ms"] == pytest.approx(1.)
    assert row["time_from_peak_ms"] == pytest.approx(.5)
    assert row["crossing_count"] == 2
    assert missing["first_time_ms"] is None and missing["crossing_count"] == 0


def test_exact_level_plateau_is_one_crossing_and_stop_is_exclusive():
    """Do not double-count samples on the level or include an endpoint crossing."""
    time = np.array([0., 1., 2., 3., 4.])
    voltage = np.array([1., 0., 0., 1., 0.])
    row = downward_crossings(time, voltage, 0., 4., (0.,))[0]
    assert row["first_time_ms"] == 1. and row["crossing_count"] == 1
