"""Tests for the per-cycle phase-plane landmarks and charge budgets."""

import numpy as np
import pytest

from docs.evidence.h01_spike_cycle_energetics import (
    charge_per_cycle, density_to_na, early_late, landmarks, phase_plane, pv_currents, slope_v_s,
    subthreshold_landmarks)


def _two_spikes():
    time = np.arange(0., 60., .02)
    voltage = np.full_like(time, -80.)
    for centre in (20., 40.):
        voltage += 100.*np.exp(-((time-centre)/.4)**2)
        voltage -= 8.*np.exp(-((time-centre-1.5)/1.)**2)
    return time, voltage


def test_landmarks_find_two_cycles_with_threshold_before_peak():
    time, voltage = _two_spikes()
    rows = landmarks(time, voltage, (10., 60.))
    assert [r["spike"] for r in rows] == [1, 2]
    first = rows[0]
    assert first["threshold_ms"] < first["peak_ms"] < first["minimum_ms"]
    assert first["threshold_mv"] < first["peak_mv"]
    assert first["max_rise_v_s"] > 0. > first["max_fall_v_s"]
    assert first["minimum_mv"] < -80.
    assert first["cycle_start_ms"] == 10.
    assert rows[1]["cycle_start_ms"] == first["minimum_ms"]
    assert rows[1]["cycle_ms"] == pytest.approx(20., abs=.1)


def test_landmarks_ignore_spikes_outside_the_pulse():
    time, voltage = _two_spikes()
    assert len(landmarks(time, voltage, (30., 60.))) == 1


def test_threshold_is_the_last_slow_sample_before_the_fastest_rise():
    time, voltage = _two_spikes()
    row = landmarks(time, voltage, (10., 60.))[0]
    slope = slope_v_s(time, voltage)
    index = int(np.flatnonzero(time == row["threshold_ms"])[0])
    assert slope[index-1] < 10. <= slope[index]
    assert row["threshold_mv"] < -60.


def test_phase_plane_returns_matching_arrays_in_the_window():
    time, voltage = _two_spikes()
    v, dv = phase_plane(time, voltage, 18., 22.)
    assert v.shape == dv.shape and len(v) == 201
    assert dv.max() == pytest.approx(slope_v_s(time, voltage)[(time >= 18.) & (time <= 22.)].max())


def test_density_conversion_uses_segment_area_and_flips_outward_sign():
    assert density_to_na(2., 150., outward_positive=True) == pytest.approx(-3.)
    assert density_to_na(2., 150., outward_positive=False) == pytest.approx(3.)


def test_pv_currents_and_charge_close_on_a_synthetic_balance():
    time = np.arange(0., 4., .01)
    geometry = {"area_um2": 100., "neighbours": [{"voltage_key": "axial_neighbour_0_mv", "resistance_mohm": 10.}]}
    voltage = np.full_like(time, -70.)
    data = {"time_ms": time, "voltage_mv": voltage, "axial_neighbour_0_mv": voltage+5.,
            "total_current_na": np.full_like(time, .2), "NaTg_current_ma_cm2": np.full_like(time, .3),
            "ina_ma_cm2": np.full_like(time, .3), "capacitive_current_ma_cm2": np.full_like(time, .4)}
    currents = pv_currents(data, geometry)
    assert set(currents) == {"applied", "NaTg", "capacitive", "axial"}
    assert currents["axial"][0] == pytest.approx(.5)
    rows = [{"spike": 1, "cycle_start_ms": 0., "minimum_ms": 2.}]
    budget = charge_per_cycle(time, currents, rows)[0]
    assert budget["applied"] == pytest.approx(.4)
    assert budget["NaTg"] == pytest.approx(-.6)
    assert budget["residual"] == pytest.approx(.4-.6-.8+1.)


def test_early_late_split_keeps_three_each_side():
    rows = [{"spike": i} for i in range(1, 8)]
    split = early_late(rows)
    assert [r["spike"] for r in split["early"]] == [1, 2, 3]
    assert [r["spike"] for r in split["late"]] == [5, 6, 7]
    assert early_late(rows[:2])["late"] == []


def test_subthreshold_landmarks_average_the_plateau_and_the_return():
    time = np.arange(0., 500., .5)
    voltage = np.where((time >= 100.) & (time < 300.), -60., -80.)
    voltage = np.where(time >= 400., -75., voltage)
    values = subthreshold_landmarks(time, voltage, (100., 300.))
    assert values["plateau_mv"] == pytest.approx(-60.)
    assert values["return_mv"] == pytest.approx(-75.)
