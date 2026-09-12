"""Regressions for SP11 temporal coverage and sampling bias."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h01_e_current_paths as paths
from h01_e_current_paths import _window_stats, lever_rule, plateau_contrast


def test_one_terminal_sample_does_not_characterize_post_pulse_window():
    result = _window_stats(np.array([2000., 2100.]), np.array([-65., -84.]), (2100., 2300.))
    assert result["status"] == "unavailable"
    assert result["mean_mv"] is None
    assert result["p50_mv"] is None


def test_adaptive_grid_mean_weights_elapsed_time():
    time = np.array([0., .01, .02, 1.])
    result = _window_stats(time, time * 10., (0., 1.))
    assert result["mean_mv"] == pytest.approx(5.)
    assert result["sample_mean_mv"] != pytest.approx(result["mean_mv"])


def test_incomplete_post_window_cannot_supply_resistance_or_current(tmp_path):
    time = np.array([1200., 1300., 1500., 1800., 2000., 2100.])
    voltage = np.array([-60., -64., -65., -65., -64., -84.])
    path = tmp_path / "human.npz"
    np.savez(path, time_ms=time, corrected_voltage_mv=voltage)
    result = plateau_contrast({"time_ms": time, "voltage_mv": voltage}, path, .2)
    assert result["model_input_resistance_mohm"] is None
    assert result["offset_na"] is None


def test_baseline_current_shares_cannot_exclude_a_causal_family():
    row = {"tail": False, "means_na": {"Nap": .01}, "shares": {"Nap": 1.}, "drift_na": .001}
    result = lever_rule({"200 pA": [row], "310 pA": [row]}, .02)
    assert result["causal_verdict"] == "not_established"
    assert result["admissible"] is None


def test_stage0_bands_compare_against_the_reference_table():
    reference = [{"peak_ms": 1100., "ahp_mv": -70.}, {"peak_ms": 1300., "ahp_mv": -71.}]
    summary = {"count": 2, "peak_ms": [1100.05, 1300.02], "trough_mv": [-70.04, -71.], "axon_first": True}
    bands = paths.stage0_bands("250 pA", summary, reference)
    assert all(b["held"] for b in bands)
    summary["peak_ms"][1] = 1300.5
    assert [b["row"] for b in paths.stage0_bands("250 pA", summary, reference) if not b["held"]] == ["peak_time_max_dev_ms"]
    summary["count"] = 3
    missed = {b["row"] for b in paths.stage0_bands("250 pA", summary, reference) if not b["held"]}
    assert missed == {"count", "peak_time_max_dev_ms", "trough_max_dev_mv"}


def _row(tail, means):
    total = sum(abs(v) for v in means.values())
    return {"tail": tail, "drift_na": .002, "means_na": dict(means, capacitive=-.002),
            "shares": {k: abs(v)/total for k, v in means.items()}}


def test_lever_rule_needs_drift_carried_and_a_smaller_high_drive_share():
    low = [_row(False, {"Nap": .01, "Im": -.001, "axial": -.005}), _row(True, {"Nap": .5, "Im": -.5, "axial": -.5})]
    high = [_row(False, {"Nap": .01, "Im": -.05, "axial": -.2})]*4
    rule = paths.lever_rule({"200 pA": low, "310 pA": high})
    assert rule["drift_200pa_na"] == .002
    assert rule["levers"]["Nap"]["carries_drift"] and rule["levers"]["Nap"]["selective_for_low_drive"]
    assert not rule["levers"]["Im"]["carries_drift"]
    assert rule["admissible"] is None
    assert rule["levers"]["Nap"]["mean_200pa_na"] == .01  # the tail is excluded from the reference


def test_window_means_and_interspike_windows():
    time = np.linspace(0., 10., 101)
    currents = {"a": np.ones_like(time), "b": time}
    means = paths.window_means(time, currents, 2., 4.)
    assert abs(means["a"]-1.) < 1e-12 and abs(means["b"]-3.) < 1e-9
    assert paths.window_means(time, currents, 2., 2.01) is None
    assert paths.window_means(time, currents, 2., 12.) is None
    rows = [{"spike": 1, "minimum_ms": 1100., "minimum_mv": -70., "threshold_ms": 1090.},
            {"spike": 2, "minimum_ms": 1300., "minimum_mv": -71., "threshold_ms": 1290.}]
    volt_t, volt = np.array([1000., 2020.]), np.array([-70., -60.])
    windows = paths.interspike_windows(rows, volt_t, volt)
    assert [w["stop_ms"] for w in windows] == [1290., 2020.] and windows[1]["tail"] and not windows[0]["tail"]


def test_plateau_offset_gates_admissibility(tmp_path):
    low = [_row(False, {"Nap": .01, "Im": -.001, "axial": -.005})]
    high = [_row(False, {"Nap": .01, "Im": -.05, "axial": -.2})]
    rule = paths.lever_rule({"200 pA": low, "310 pA": high}, offset_na=.05)
    assert rule["levers"]["Nap"]["carries_drift"] and not rule["levers"]["Nap"]["carries_plateau_offset"]
    assert rule["admissible"] is None
    time = np.arange(0., 2400., .1)
    human = np.full_like(time, -84.)
    human[(time >= 1020.) & (time < 2020.)] = -67.
    np.savez(tmp_path/"sweep-56.npz", time_ms=time, corrected_voltage_mv=human)
    model = np.full_like(time, -84.)
    model[(time >= 1020.) & (time < 2020.)] = -62.
    contrast = paths.plateau_contrast({"time_ms": time, "voltage_mv": model}, tmp_path/"sweep-56.npz", .2)
    assert abs(contrast["late_pulse_offset_mv"]-5.) < 1e-9
    assert contrast["model_input_resistance_mohm"] is None
    assert contrast["offset_na"] is None
