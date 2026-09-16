"""Synthetic-sweep tests for the C3 human long-square datums."""

import json

import numpy as np
import pytest

from h01_c3_human_datums import (DETECTION_MV, JUNCTION_MV, count_band, derive_datums, export_counts, main,
                                 measure, read_long_square_sweeps, render, return_window, stimulus_window,
                                 sweep_features)

RATE_HZ = 10000.


def _command(amplitude_pa, on_ms=1020., off_ms=2020., duration_ms=2500., test_pulse_pa=50., holding_pa=3.):
    time = np.arange(int(duration_ms*RATE_HZ/1e3))/RATE_HZ*1e3
    current = np.full_like(time, holding_pa)
    current[(time >= 5.) & (time < 15.)] += test_pulse_pa          # the protocol's short test pulse
    current[(time >= on_ms) & (time < off_ms)] += amplitude_pa
    return time, current


def _voltage(time, spike_times_ms, rest_mv=-70., after_mv=None):
    voltage = np.full_like(time, rest_mv)
    if after_mv is not None:
        voltage[time >= 2020.] = after_mv
    for centre in spike_times_ms:
        voltage[(time >= centre) & (time < centre+1.)] = 20.
    return voltage


def test_stimulus_window_is_the_longest_excursion_not_the_test_pulse():
    time, current = _command(-10.)
    window = stimulus_window(time, current)
    assert window["amplitude_pa"] == pytest.approx(-10.) and window["baseline_pa"] == pytest.approx(3.)
    assert window["on_ms"] == pytest.approx(1020.) and window["off_ms"] == pytest.approx(2020.)
    time, current = _command(250.)
    assert stimulus_window(time, current)["amplitude_pa"] == pytest.approx(250.)
    flat = stimulus_window(time, np.full_like(time, 3.))
    assert flat["on_ms"] is None and flat["amplitude_pa"] == 0.


def test_sweep_features_counts_only_inside_the_window_and_reads_the_rest():
    time, current = _command(200.)
    voltage = _voltage(time, [10., 1100., 1500., 1900., 2100.])     # test-pulse, three inside, one after
    row = sweep_features(time, voltage, current)
    assert row["count"] == 3 and row["amplitude_pa"] == pytest.approx(200.)
    assert row["rest_mv"] == pytest.approx(-70.) and row["rest_sd_mv"] == pytest.approx(0.)
    assert row["after_mv"] == pytest.approx(-70.) and row["after_available"] is True
    attributed = sweep_features(time, voltage, current, amplitude_attr_pa=199.999)
    assert attributed["amplitude_pa"] == 199.999 and attributed["measured_amplitude_pa"] == pytest.approx(200.)
    flat = sweep_features(time, voltage, np.full_like(time, 3.))
    assert flat["count"] is None and flat["rest_mv"] is None and flat["after_mv"] is None
    settled = sweep_features(time, _voltage(time, [], after_mv=-75.), current)
    assert settled["after_mv"] == pytest.approx(-75.) and settled["rest_mv"] == pytest.approx(-70.)


def test_return_window_is_the_runners_ten_ms_ending_200_ms_after_offset():
    from h01_anatomy_transfer_run import windows
    time = np.arange(0., 2500., .1)
    voltage = np.full_like(time, -70.)
    voltage[(time >= 2210.) & (time <= 2220.)] = -80.       # exactly the runner's samples
    voltage[(time > 2220.) | ((time >= 2200.) & (time < 2210.))] = 0.
    ours = return_window(time, voltage, 2020.)
    theirs = windows(time, voltage, (1020., 2020.))
    assert ours["mean_mv"] == pytest.approx(theirs["return_mv"]) == pytest.approx(-80.)
    assert ours["window_ms"] == [2210., 2220.] and ours["available"] is True and ours["n"] == 101
    short = return_window(time[time < 2100.], voltage[time < 2100.], 2020.)
    assert short["mean_mv"] is None and short["available"] is False
    assert windows(time[time < 2100.], voltage[time < 2100.], (1020., 2020.))["return_available"] is False


def _family():
    rows = []
    for sweep, (amplitude, count) in enumerate([(-50., 0), (10., 0), (30., 0), (50., 0), (70., 2), (90., 5),
                                                (110., 8), (130., 11), (150., 0)], start=20):
        rows.append(dict(sweep=sweep, amplitude_pa=amplitude, count=count, rest_mv=-70.+.1*(sweep % 3),
                         after_mv=-72.+.1*(sweep % 3), after_available=True))
    rows += [dict(sweep=30+i, amplitude_pa=90., count=c, rest_mv=-70., after_mv=-72., after_available=True)
             for i, c in enumerate((5, 6, 4))]
    rows.append(dict(sweep=33, amplitude_pa=40., count=None, rest_mv=None))    # no window: ignored
    return rows


def test_derive_datums_reads_the_firing_edges_and_the_family_step():
    datums = derive_datums(_family(), 90., [5, 5, 4])
    assert datums["rheobase_pa"] == 70. and datums["rheobase_count"] == 2
    assert datums["highest_firing_pa"] == 130. and datums["highest_firing_count"] == 11
    assert datums["highest_recorded_pa"] == 150. and datums["sweep_step_pa"] == 20.   # repeats do not shift it
    assert datums["measured_counts_at_primary"] == [5, 5, 6, 4] and datums["measured_sweeps_at_primary"] == [25, 30, 31, 32]
    assert datums["repeat_counts"] == [5, 5, 4] and datums["primary_pa"] == 90.
    assert datums["family_amplitudes_pa"] == [10., 30., 50., 70., 90., 110., 130., 150.]
    assert datums["rest_repeat_sd_mv"] > 0. and datums["rest_repeat_mean_mv"] == pytest.approx(-69.925)
    assert datums["after_repeat_mean_mv"] == pytest.approx(-71.925) and datums["after_repeat_sd_mv"] > 0.
    assert datums["after_sweeps_unreached"] == []
    band = datums["count_band"]     # 70, 90 x4, 110 pA: one 20 pA step around the 90 pA primary
    assert (band["min"], band["max"]) == (2, 8) and band["counts"] == [2, 5, 5, 6, 4, 8]
    assert band["sweeps"] == [24, 25, 30, 31, 32, 26] and band["amplitudes_pa"] == [70., 90., 90., 90., 90., 110.]


def test_count_band_tolerates_the_recorded_amplitude_jitter_and_needs_a_step():
    positive = [dict(sweep=1, amplitude_pa=289.999996, count=9), dict(sweep=2, amplitude_pa=309.999998, count=10),
                dict(sweep=3, amplitude_pa=330.0000005, count=12), dict(sweep=4, amplitude_pa=350.000001, count=13)]
    band = count_band(positive, 310., 20.000001654807022)
    assert band["counts"] == [9, 10, 12] and (band["min"], band["max"]) == (9, 12) and band["sweeps"] == [1, 2, 3]
    assert count_band(positive, 310., None)["min"] is None
    assert count_band([], 310., 20.)["counts"] == []


def test_derive_datums_records_sweeps_that_do_not_reach_the_return_window():
    rows = [dict(sweep=1, amplitude_pa=90., count=3, rest_mv=-70., after_mv=None, after_available=False),
            dict(sweep=2, amplitude_pa=110., count=5, rest_mv=-70., after_mv=-71., after_available=True)]
    datums = derive_datums(rows, 90., [3])
    assert datums["after_sweeps_unreached"] == [1] and datums["after_repeat_mean_mv"] == -71.
    assert datums["after_repeat_sd_mv"] is None    # one supporting sweep: no repeat spread
    none = derive_datums([rows[0]], 90., [3])
    assert none["after_repeat_mean_mv"] is None and none["after_sweeps_unreached"] == [1]


def test_derive_datums_without_firing_sweeps_leaves_the_edges_empty():
    rows = [dict(sweep=1, amplitude_pa=10., count=0, rest_mv=-70.)]
    datums = derive_datums(rows, 10., [0])
    assert datums["rheobase_pa"] is None and datums["highest_firing_pa"] is None
    assert datums["sweep_step_pa"] is None and datums["rest_repeat_sd_mv"] is None
    assert derive_datums([], 10., [])["highest_recorded_pa"] is None


def _write_nwb(path, sweeps, conversion=1e-12, in_amps=True):
    import h5py
    with h5py.File(path, "w") as handle:
        for sweep, (kind, amplitude, spikes) in sweeps.items():
            time, current = _command(amplitude)
            stimulus = handle.create_group(f"stimulus/presentation/Sweep_{sweep}")
            stimulus["aibs_stimulus_name"] = np.bytes_(kind)
            stimulus["aibs_stimulus_amplitude_pa"] = amplitude+1e-6
            data = stimulus.create_dataset("data", data=current*1e-12 if in_amps else current)
            data.attrs["conversion"] = conversion
            stimulus.create_dataset("starting_time", data=0.).attrs["rate"] = RATE_HZ
            acquisition = handle.create_group(f"acquisition/timeseries/Sweep_{sweep}")
            voltage = _voltage(time, spikes)-JUNCTION_MV          # stored uncorrected, in volts
            acquisition.create_dataset("data", data=voltage*1e-3).attrs["conversion"] = .001
            acquisition.create_dataset("starting_time", data=0.).attrs["rate"] = RATE_HZ


def test_read_long_square_sweeps_applies_the_junction_correction_and_skips_other_stimuli(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path/"cell.nwb"
    _write_nwb(path, {7: ("Long Square", 90., [1100., 1300.]), 8: ("Short Square", 500., [1100.]),
                      9: ("Long Square", -30., [])})
    rows = read_long_square_sweeps(path)
    assert [r["sweep"] for r in rows] == [7, 9] and rows[0]["stimulus"] == "Long Square"
    assert rows[0]["count"] == 2 and rows[0]["amplitude_pa"] == pytest.approx(90., abs=1e-5)
    assert rows[0]["measured_amplitude_pa"] == pytest.approx(90.) and rows[0]["rest_mv"] == pytest.approx(-70.)
    assert rows[1]["count"] == 0 and rows[1]["on_ms"] == pytest.approx(1020.)


def test_export_counts_reads_the_donor_exports_in_their_own_convention(tmp_path):
    (tmp_path/"human-pv").mkdir()
    time = np.arange(0., 1500., .1)
    np.savez(tmp_path/"human-pv"/"active-0.npz", time=time, voltage=_voltage(time, [300., 400., 1300.]))
    rows = export_counts(tmp_path, [("human-pv/active-0.npz", .19)], (270., 1270.))
    assert rows[0]["count"] == 2 and rows[0]["amplitude_pa"] == pytest.approx(190.) and len(rows[0]["sha256"]) == 64


def test_measure_and_main_write_one_entry_per_donor(tmp_path, monkeypatch, capsys):
    pytest.importorskip("h5py")
    (tmp_path/"cells").mkdir()
    _write_nwb(tmp_path/"cells"/"a.nwb", {1: ("Long Square", 50., []), 2: ("Long Square", 70., [1100.]),
                                          3: ("Long Square", 90., [1100., 1200.])})
    donors = {"donor-a": dict(nwb="cells/a.nwb", primary_pa=90., repeat_counts=[2], source_cell="synthetic")}
    report = measure(tmp_path, donors)
    entry = report["donors"]["donor-a"]
    assert entry["rheobase_pa"] == pytest.approx(70., abs=1e-5) and entry["highest_firing_pa"] == pytest.approx(90., abs=1e-5)
    assert entry["sweep_step_pa"] == pytest.approx(20., abs=1e-5) and entry["measured_counts_at_primary"] == [2]
    assert entry["crossing_threshold_mv"] == DETECTION_MV and len(entry["sha256"]) == 64 and "donor_exports" not in entry
    assert entry["count_band"]["counts"] == [1, 2] and entry["count_band"]["sweeps"] == [2, 3]
    assert entry["after_repeat_mean_mv"] == pytest.approx(-70.) and entry["after_sweeps_unreached"] == []
    assert "donor-a" in render(report) and "[1, 2] [2, 3]" in render(report)
    monkeypatch.setattr("h01_c3_human_datums.DONORS", donors)
    output = tmp_path/"out"/"human-datums.json"
    main(["--cache", str(tmp_path), "--output", str(output)])
    written = json.loads(output.read_text())
    assert set(written["donors"]) == {"donor-a"} and "definition" in written
    assert "donor-a" in capsys.readouterr().out
