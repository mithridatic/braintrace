"""Protect current discontinuities and reject ambiguous event recordings."""

import numpy as np
import pytest

from docs.evidence.h01_l2_recording import right_limit_recording


def test_event_duplicates_keep_right_limit_and_raw_indices():
    time = np.array([0., 1., 1., 1., 2.])
    voltage = np.array([-70., -65., -65., -65., -60.])
    current = np.array([0., 0., .1, .2, .2])
    assert not np.all(np.diff(time) > 0)
    result = right_limit_recording(time, voltage, current)
    np.testing.assert_array_equal(result["raw_sample_index"], [0, 3, 4])
    np.testing.assert_array_equal(result["applied_current_na"], [0., .2, .2])
    np.testing.assert_array_equal(result["voltage_mv"], [-70., -65., -60.])
    np.testing.assert_array_equal(time, [0., 1., 1., 1., 2.])
    assert np.all(np.diff(result["time_ms"]) > 0)


def test_no_duplicates_preserves_every_sample():
    result = right_limit_recording([0., 1., 2.], [-70., -60., -65.], [0., .1, 0.])
    np.testing.assert_array_equal(result["raw_sample_index"], [0, 1, 2])


@pytest.mark.parametrize("time,voltage,current", [
    ([0., 1., .5], [-70.]*3, [0.]*3),
    ([0., 1., 1.], [-70., -60., -59.], [0.]*3),
    ([0., 0.], [-70.]*2, [0.]*2),
    ([0., 1.], [-70., np.nan], [0.]*2),
    ([0., 1.], [-70.]*2, [0., np.inf]),
    ([0., 1.], [-70.], [0.]*2),
    ([[0., 1.]], [[-70., -60.]], [[0., 0.]]),
])
def test_invalid_recording_is_not_silently_repaired(time, voltage, current):
    with pytest.raises(ValueError):
        right_limit_recording(time, voltage, current)
