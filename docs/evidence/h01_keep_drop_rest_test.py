"""Tests for the pure rest-window measurement of the keep/drop rule."""

import numpy as np
import pytest

from docs.evidence.h01_keep_drop_rest import DONORS, rest_window


def _trace(duration_ms=200., dt=.02, level=-80.):
    time = np.arange(0., duration_ms, dt)
    return time, np.full_like(time, level)


def test_flat_trace_rest_is_the_level():
    time, voltage = _trace()
    rest = rest_window(time, voltage, 120.)
    assert rest["mean_mv"] == pytest.approx(-80.)
    assert rest["sd_mv"] == 0.
    assert rest["n"] == 5000
    assert rest["window_ms"] == [20., 120.]


def test_offset_outside_the_window_is_ignored():
    time, voltage = _trace()
    voltage[time >= 120.] = 20.      # pulse response after onset
    voltage[time < 20.] = -40.       # earlier junk before the window
    rest = rest_window(time, voltage, 120.)
    assert rest["mean_mv"] == pytest.approx(-80.)
    assert rest["sd_mv"] == 0.


def test_empty_window_raises():
    time, voltage = _trace(duration_ms=50.)
    with pytest.raises(ValueError):
        rest_window(time, voltage, 500.)


def test_donor_table_names_four_deployed_donors():
    assert set(DONORS) == {"l2-pyramidal-allen-541563728", "l4-pyramidal-allen-527952884",
                           "l3-sst-interneuron-hl5mn1", "l5-pv-basket-hl5bn1"}
    assert all(spec["shift_mv"] == -14. for key, spec in DONORS.items() if spec["source"].endswith(".nwb"))
