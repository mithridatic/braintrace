"""Tests for the row-by-row contract scorer."""

import numpy as np
import pytest

from docs.evidence.h01_contract_score import (attach_limits, dixon_limits, event_rows, recovery_rows,
                                              subthreshold_rows)
from docs.evidence.h01_pv_human_datums import spike_datums


def _trace(peaks_ms, shift_ms=0., duration_ms=60., dt=.01):
    time = np.arange(0., duration_ms, dt)
    voltage = np.full(time.shape, -70.)
    for peak in peaks_ms:
        voltage += 100.*np.exp(-((time-peak-shift_ms)/.4)**2)
        voltage -= 8.*np.exp(-((time-peak-shift_ms-2.)/1.5)**2)
    return time, voltage


def _human(peaks_ms):
    time, voltage = _trace(peaks_ms)
    return spike_datums(time, voltage)


def test_identical_trace_passes_every_row():
    rows, events = event_rows(*_trace((10., 20., 30.)), _human((10., 20., 30.)), (0., 60.))
    assert len(events) == 3
    assert {r["verdict"] for r in rows} == {"pass"}
    assert all(r["residual"] == 0. for r in rows if r["kind"] != "count")


def test_shift_gives_the_shift_as_residual_and_fails_timing_rows():
    rows, _ = event_rows(*_trace((10., 20.), shift_ms=1.5), _human((10., 20.)), (0., 60.))
    by_key = {r["key"]: r for r in rows}
    assert by_key["e1_rise_crossing_ms"]["residual"] == pytest.approx(1.5, abs=.02)
    assert by_key["e1_rise_crossing_ms"]["verdict"] == "fail"
    assert by_key["e1_peak_sample_voltage_mv"]["verdict"] == "pass"
    assert by_key["i1_ms"]["derived"] and by_key["i1_ms"]["verdict"] == "pass"


def test_missing_event_fails_count_and_marks_rows_unavailable():
    rows, _ = event_rows(*_trace((10.,)), _human((10., 20.)), (0., 60.))
    by_key = {r["key"]: r for r in rows}
    assert by_key["event_count"]["verdict"] == "fail"
    assert by_key["e2_rise_crossing_ms"]["verdict"] == "unavailable"
    assert by_key["e2_rise_crossing_ms"]["residual"] is None
    assert by_key["i1_ms"]["verdict"] == "unavailable"


def test_tiers_follow_the_ratio():
    rows, _ = event_rows(*_trace((10., 20.), shift_ms=12.), _human((10., 20.)), (0., 60.))
    by_key = {r["key"]: r for r in rows}
    assert by_key["e1_rise_crossing_ms"]["tier"] == "target"
    rows, _ = event_rows(*_trace((10., 20.), shift_ms=2.5), _human((10., 20.)), (0., 60.))
    assert {r["key"]: r for r in rows}["e1_rise_crossing_ms"]["tier"] == "real"


def test_recovery_rows_compare_minimum_and_delay():
    time, voltage = _trace((10., 20.))
    events = spike_datums(time, voltage)
    datums = [{"event_index": 0, "minimum_sample_voltage_mv": -75., "delay_from_peak_ms": 2., "terminal_window": False},
              {"event_index": 1, "minimum_sample_voltage_mv": -75., "delay_from_peak_ms": 2., "terminal_window": True},
              {"event_index": 2, "minimum_sample_voltage_mv": -75., "delay_from_peak_ms": 2., "terminal_window": False}]
    rows = recovery_rows(time, voltage, events, datums, 60.)
    by_key = {r["key"]: r for r in rows}
    assert by_key["m1_voltage_mv"]["model"] < -70.
    assert by_key["m1_delay_ms"]["residual"] == pytest.approx(0., abs=.3)
    assert by_key["m2_voltage_mv"]["verdict"] == "unavailable"
    assert by_key["m3_voltage_mv"]["verdict"] == "unavailable" and by_key["m3_voltage_mv"]["model"] is None


def test_subthreshold_rows_interpolate_and_refuse_extrapolation():
    time = np.arange(0., 10., .5)
    voltage = -80.+time
    rows = subthreshold_rows(time, voltage, [{"requested_time_ms": 2.25, "voltage_mv": -77.},
                                             {"requested_time_ms": 20., "voltage_mv": -60.}])
    assert rows[0]["residual"] == pytest.approx(-.75)
    assert rows[0]["verdict"] == "pass"
    assert rows[1]["verdict"] == "unavailable"
    with pytest.raises(ValueError):
        subthreshold_rows(time[::-1], voltage, [])
    with pytest.raises(ValueError):
        subthreshold_rows(time, np.where(time > 5., np.nan, voltage), [])


def test_dixon_limits_use_the_range_times_the_table_factor():
    a = {"x": 1., "y": None, "z": 3.}
    b = {"x": 1.2, "y": 2., "z": 3.}
    out = dixon_limits([a, b])
    assert out["factor"] == 2.95
    assert out["limits"]["x"] == pytest.approx(.2*2.95)
    assert out["limits"]["z"] == 0.
    assert "y" not in out["limits"]
    assert dixon_limits([a, b, b])["factor"] == 1.81
    with pytest.raises(ValueError):
        dixon_limits([a])


def test_attach_limits_turns_noise_level_failures_into_unresolved():
    rows = [{"key": "k", "residual": .3, "verdict": "fail"}, {"key": "p", "residual": .1, "verdict": "pass"},
            {"key": "big", "residual": 5., "verdict": "fail"}]
    out = attach_limits(rows, {"k": .5, "p": .5, "big": .5})
    assert [r["verdict"] for r in out] == ["unresolved", "pass", "fail"]
    assert attach_limits([{"key": "k", "residual": .3, "verdict": "fail"}], None)[0]["numerical_limit"] is None
