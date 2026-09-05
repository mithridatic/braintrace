"""Reject vacuous, shifted, missing, and clipped spike comparisons."""
import numpy as np
import pytest
from docs.evidence.h01_ei_spike_transfer import compare_spike_transfer


def trace(shift=0., peak=10.):
    return dict(time_ms=np.array([0., 1., 1.1, 1.2, 1.3, 2., 3.])+shift,
                voltage_mv=np.array([-80., -80., peak, 0., -80., -80., -80.]))


def test_identical_nonzero_events_pass():
    report = compare_spike_transfer(trace(), trace(), start_ms=.5, stop_ms=2.5)
    assert report["passed"] and len(report["signed_errors"]) == 1


@pytest.mark.parametrize("kind", ["empty", "peak", "time", "missing"])
def test_wrong_direct_response_fails(kind):
    ref, actual = trace(), trace()
    if kind == "empty":
        ref["voltage_mv"][:] = -80.
        actual["voltage_mv"][:] = -80.
    if kind == "missing": actual["voltage_mv"][:] = -80.
    if kind == "peak": actual = trace(peak=12.)
    if kind == "time": actual = trace(shift=.2)
    assert not compare_spike_transfer(ref, actual, start_ms=.5, stop_ms=2.5)["passed"]


@pytest.mark.parametrize("kind", ["nan", "order", "coverage", "clipped"])
def test_invalid_trace_is_not_a_negative_causal_result(kind):
    actual = trace()
    if kind == "nan": actual["voltage_mv"][0] = np.nan
    if kind == "order": actual["time_ms"][2] = 0.
    if kind == "coverage": actual["time_ms"][-1] = 2.1
    if kind == "clipped": actual["voltage_mv"][-2] = 0.
    with pytest.raises(ValueError):
        compare_spike_transfer(trace(), actual, start_ms=.5, stop_ms=2.5)
