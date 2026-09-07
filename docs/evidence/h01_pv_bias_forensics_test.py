"""Tests for the I input datum forensic split."""

import numpy as np
import pytest

from docs.evidence.h01_pv_bias_forensics import (
    assemble, held_sweeps, baseline_slope_mohm, input_resistance_mohm, model_rest_mv, render,
    repeat_limit_mv, verdict)


def _records():
    rows = [(20, "Long Square", 25., -110., -87.5, -95.2), (23, "Long Square", 28., -50., -88., -91.7),
            (35, "Long Square", 31., 190., -86.9, -65.7), (39, "Long Square", 31., 270., -86.9, -67.9),
            (40, "Long Square", 31., 120., -87., -70.), (41, "Long Square", 31., 120., -86.8, -70.),
            (42, "Long Square", 31., 120., -86.4, -70.), (53, "Square - 0.5ms Subthreshold", -21., 200., -87.9, None)]
    records = []
    for sweep, name, bias, command, base, late in rows:
        record = {"sweep": sweep, "name": name, "bias_pa": bias, "command_pa": command,
                  "baseline_mv": base, "baseline_sd_mv": .1}
        if late is not None:
            record["late_pulse_mv"] = late
        records.append(record)
    return records


def test_input_resistance_uses_only_hyperpolarising_passive_sweeps():
    value = input_resistance_mohm(_records())
    assert value == pytest.approx(np.median([7.7/110.*1000., 3.7/50.*1000.]))


def test_repeat_limit_applies_dixon_factor_for_three_repeats():
    assert repeat_limit_mv(_records()) == pytest.approx(.6*1.81)


def test_baseline_slope_is_far_below_input_resistance_for_a_tracking_hold():
    assert abs(baseline_slope_mohm(_records())) < 40.


@pytest.mark.parametrize("rests, expected", [
    ({"bias_0": -87., "bias_measured": -84.5}, "held state"),
    ({"bias_0": -89.5, "bias_measured": -87.}, "membrane current"),
    ({"bias_0": -87., "bias_measured": -86.5}, "undecided"),
])
def test_verdict_names_the_input_inside_the_limit(rests, expected):
    assert verdict(-86.9, 1., rests).startswith(expected)


def test_assemble_reports_rest_errors_and_model_resistance():
    report = assemble(_records(), {"bias_0": -87., "bias_measured": -84.5})
    assert report["human_held_baseline_mv"] == pytest.approx(-86.9)
    assert report["model_rest_error_mv"]["bias_measured"] == pytest.approx(2.4)
    assert report["model_input_resistance_mohm"] == pytest.approx(2.5/31.*1000.)
    assert report["predicted_shift_if_current_mv"] == pytest.approx(report["input_resistance_mohm"]*.031)
    assert "Verdict" in render(report)


def test_model_rest_averages_the_pre_pulse_window(tmp_path):
    time = np.arange(0., 300., .5)
    voltage = np.where(time < 270., -87., 0.)
    path = tmp_path/"trace.npz"
    np.savez(path, time_ms=time, voltage_mv=voltage)
    assert model_rest_mv(path) == pytest.approx(-87.)


def test_held_sweeps_skips_noise_sweeps_so_the_sealed_holdout_is_unread(tmp_path):
    import h5py
    with h5py.File(tmp_path/"cell.nwb", "w") as nwb:
        for sweep, name in ((40, "Long Square"), (48, "Noise 1"), (44, "Noise 1")):
            group = nwb.create_group(f"acquisition/timeseries/Sweep_{sweep}")
            group.create_dataset("data", data=np.full(60000, -.073, dtype=np.float32))
            group.create_dataset("aibs_stimulus_name", data=name.encode())
            group.create_dataset("bias_current", data=3.1e-11)
            nwb.create_dataset(f"stimulus/presentation/Sweep_{sweep}/data", data=np.full(60000, 1.2e-10))
        records = held_sweeps(nwb)
    assert [r["sweep"] for r in records] == [40]
