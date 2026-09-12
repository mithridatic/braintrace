"""Physical and temporal edge cases for the direct observation contract."""

import numpy as np
import pytest

from docs.evidence.h01_direct_observations import (
    first_crossing_ms,
    observe,
    select_windows,
    validate_trace,
    window_stats,
)


def spike_trace(count=4):
    t = np.arange(0., 121., .05)
    v = np.full_like(t, -70.)
    for centre in np.arange(count)*20+15:
        v += 100*np.exp(-((t-centre)/.4)**2)-8*np.exp(-((t-centre-2)/1.5)**2)
    return t, v


@pytest.mark.parametrize("t,v", [([0], [0]), ([0, 1], [0]), ([[0, 1]], [[0, 1]]),
    ([0, 0], [1, 2]), ([1, 0], [1, 2]), ([0, np.nan], [0, 1]), ([0, 1], [0, np.inf])])
def test_invalid_clocks_and_values_are_rejected(t, v):
    with pytest.raises(ValueError):
        validate_trace(t, v)


def test_covered_sparse_window_interpolates_endpoints_without_extrapolation():
    r = window_stats([0., 2.], [0., 20.], (.5, 1.5))
    assert r["status"] == "available" and r["mean_mv"] == pytest.approx(10.)
    assert r["p50_mv"] is None
    assert window_stats([0., 2.], [0., 20.], (1., 3.))["mean_mv"] is None


@pytest.mark.parametrize("count", [0, 1, 2, 4, 6])
def test_small_nested_sample_retains_available_event_identities(count):
    t, v = spike_trace(count)
    windows, rows = select_windows(t, v, (0., 120.))
    assert len(rows) == count
    assert len([w for w in windows if w["kind"] == "recovery"]) == min(3, max(0, count-1))
    assert len([w for w in windows if w["kind"] == "spike"]) == min(3, count)
    if count:
        assert windows[-1]["kind"] == "tail" and windows[-1]["cycle"] == count
    else:
        assert windows[0]["kind"] == "no_spike_pulse"


def test_uncovered_pulse_and_tail_are_explicit():
    t, v = spike_trace(1)
    record, arrays = observe(t, v, (0., 130.))
    assert record["windows"][-1]["status"] == "unavailable"
    assert f"{record['windows'][-1]['array_prefix']}_time_ms" not in arrays
    windows, _ = select_windows(t, v, (200., 300.))
    assert windows[0]["status"] == "unavailable"
    with pytest.raises(ValueError):
        select_windows(t, v, (10., 0.))


def test_complete_spikes_have_threshold_peak_and_trough_samples():
    t, v = spike_trace(2)
    record, arrays = observe(t, v, (0., 120.), offsets_ms=(0., 2., 999.))
    spike = next(w for w in record["windows"] if w["kind"] == "spike")
    assert [s["phase"] for s in spike["phase_samples"]] == ["threshold", "peak", "trough"]
    assert spike["phase_samples"][1]["voltage_mv"] > 0
    recovery = next(w for w in record["windows"] if w["kind"] == "recovery")
    assert recovery["phase_samples"][-1]["status"] == "beyond_window"
    assert recovery["phase_samples"][-1]["voltage_mv"] is None
    for w in record["windows"]:
        selected = arrays[f"{w['array_prefix']}_time_ms"]
        original = t[(t > w["start_ms"]) & (t < w["stop_ms"])]
        np.testing.assert_array_equal(selected[1:-1], original)


def test_equal_means_do_not_erase_opposite_trajectories():
    t = np.array([0., .25, .5, .75, 1.])
    a, aa = observe(t, -70+t, (0., 1.))
    b, bb = observe(t, -69-t, (0., 1.))
    assert aa["w0_voltage_mv"].mean() == pytest.approx(bb["w0_voltage_mv"].mean())
    assert not np.array_equal(aa["w0_voltage_mv"], bb["w0_voltage_mv"])
    assert a["causal_verdict"] == b["causal_verdict"] == "not_established"
    assert a["current_evidence"] == "unavailable"


def test_signed_charge_is_integrated_on_the_actual_clock():
    t = np.array([0., .01, .1, 1.])
    record, arrays = observe(t, -70+t, (0., 1.), {"inward": t, "outward": -t})
    assert arrays["w0_charge_inward_pc"][-1] == pytest.approx(.5)
    assert arrays["w0_charge_outward_pc"][-1] == pytest.approx(-.5)
    assert record["windows"][0]["phase_samples"][0]["currents_na"]["inward"] == 0
    for currents in ({"bad": [1.]}, {"bad": [1., 2., np.nan, 4.]}):
        with pytest.raises(ValueError):
            observe(t, -70+t, (0., 1.), currents)
    with pytest.raises(ValueError):
        observe(t, -70+t, (0., 1.), offsets_ms=(-1.,))


def test_crossing_reuses_existing_complete_event_definition():
    t, v = spike_trace(1)
    assert 14 < first_crossing_ms(t, v, 0.) < 15
    assert first_crossing_ms(t, v, 100.) is None


def test_spike_cut_off_by_pulse_end_is_explicitly_incomplete():
    t, v = spike_trace(1)
    windows, rows = select_windows(t, v, (0., 15.1))
    assert not rows
    assert any(w["kind"] == "incomplete_spike" and w["status"] == "unavailable" for w in windows)
