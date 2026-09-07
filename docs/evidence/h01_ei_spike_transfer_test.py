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


def gaussian_spike(step, centre=281.20317, sigma=.3, peak=20., start=270., stop=300.):
    """End-of-step samples (first at start+step) of a spike whose analytic peak is known."""
    t = np.arange(start+step, stop+step/2., step)
    return dict(time_ms=t, voltage_mv=-80.+(peak+80.)*np.exp(-((t-centre)/sigma)**2))


@pytest.mark.parametrize("step", [.005, .0025])
def test_interpolated_peak_recovers_the_analytic_peak_at_both_sampling_steps(step):
    from docs.evidence.h01_ei_spike_transfer import interpolated_peak_voltage
    spike = gaussian_spike(step)
    report = compare_spike_transfer(spike, spike, start_ms=275., stop_ms=295., peak_method="interpolated")
    event = report["actual_events"][0]
    assert abs(event["peak_interpolated_voltage_mv"]-20.) < 1e-3
    assert event["peak_sample_voltage_mv"] <= event["peak_interpolated_voltage_mv"]
    v = spike["voltage_mv"]
    assert interpolated_peak_voltage(spike["time_ms"], v, int(np.argmax(v))) == event["peak_interpolated_voltage_mv"]


def test_interpolated_peak_is_exact_for_a_parabola_on_a_nonuniform_grid():
    from docs.evidence.h01_ei_spike_transfer import interpolated_peak_voltage
    t = np.array([0., .3, .7, 1.2, 1.25, 1.31, 1.9, 2.6])
    v = 15.-40.*(t-1.28)**2
    assert interpolated_peak_voltage(t, v, int(np.argmax(v))) == pytest.approx(15., abs=1e-9)
    flat = np.array([1., 1., 1.])
    assert interpolated_peak_voltage(np.array([0., 1., 2.]), flat, 1) == 1.
    outside = np.array([0., 1., 1.5])  # concave, vertex at t 2.5: beyond the span, so the sample is kept
    assert interpolated_peak_voltage(np.array([0., 1., 2.]), outside, 1) == 1.


def test_peak_method_selects_the_gated_peak_and_both_are_reported():
    ref, actual = gaussian_spike(.005), gaussian_spike(.0025)
    sample = compare_spike_transfer(ref, actual, start_ms=275., stop_ms=295.)
    interpolated = compare_spike_transfer(ref, actual, start_ms=275., stop_ms=295., peak_method="interpolated")
    assert sample["peak_method"] == "sample" and "peak_sample_voltage_mv" in sample["limits"]
    assert interpolated["peak_method"] == "interpolated" and "peak_interpolated_voltage_mv" in interpolated["limits"]
    assert "peak_sample_voltage_mv" not in interpolated["limits"]
    for report in (sample, interpolated):
        assert {"peak_sample_voltage_mv", "peak_interpolated_voltage_mv"} <= set(report["signed_errors"][0])
    assert abs(interpolated["signed_errors"][0]["peak_interpolated_voltage_mv"]) < abs(
        sample["signed_errors"][0]["peak_sample_voltage_mv"])
    with pytest.raises(ValueError, match="peak_method"):
        compare_spike_transfer(ref, actual, start_ms=275., stop_ms=295., peak_method="max")
