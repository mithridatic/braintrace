"""Tests for the pure scoring parts of the anatomy-transfer runner."""

import numpy as np
import pytest

from h01_anatomy_transfer_run import count_verdict, crossings


def _trace(spike_times_ms, duration_ms=100., dt=.01):
    time = np.arange(dt, duration_ms+dt/2, dt)
    voltage = np.full_like(time, -70.)
    for centre in spike_times_ms:
        voltage[(time >= centre) & (time < centre+.5)] = 20.
    return time, voltage


def test_crossings_counts_only_inside_the_pulse():
    time, voltage = _trace([5., 25., 45., 80.])
    observed = crossings(time, voltage, (20., 60.))
    assert observed["count"] == 2
    assert observed["finite"] is True
    assert observed["first_spike_ms"] == pytest.approx(5., abs=.02)
    assert len(observed["crossing_times_ms"]) == 2


def test_crossings_reports_nonfinite_without_raising():
    time, voltage = _trace([30.])
    voltage[-10:] = np.nan
    observed = crossings(time, voltage, (20., 60.))
    assert observed["finite"] is False
    assert observed["count"] == 1


def test_crossings_empty_pulse_window():
    time, voltage = _trace([30.])
    observed = crossings(time, voltage, (200., 300.))
    assert observed == dict(count=0, first_spike_ms=None, finite=True, crossing_times_ms=[])


@pytest.mark.parametrize("model, registered, repeats, exact, band", [
    (10, 10, None, "held", None),
    (11, 10, None, "missed_within_one", None),
    (13, 10, None, "rejected", None),
    (13, 14, (14, 14, 13, 12), "missed_within_one", "held"),
    (16, 14, (14, 14, 13, 12), "rejected", "missed"),
])
def test_count_verdict(model, registered, repeats, exact, band):
    assert count_verdict(model, registered, repeats) == dict(exact=exact, repeat_band=band)
