"""Tests for the per-cycle charge budget report."""

import json

import numpy as np
import pytest

from docs.evidence.h01_energetic_report import budgets, instants, main, shares, windows


def _row():
    return {"spike": 1, "cycle_start_ms": 0., "threshold_ms": 1., "peak_ms": 2., "minimum_ms": 4.,
            "cycle_ms": 4., "minimum_mv": -75., "peak_mv": 20., "max_rise_v_s": 500., "threshold_mv": -60.}


def test_windows_cover_upstroke_falling_and_cycle():
    assert windows(_row()) == {"upstroke": (1., 2.), "falling": (2., 4.), "cycle": (0., 4.)}
    row = {**_row(), "threshold_ms": None}
    assert windows(row)["upstroke"] == (0., 2.)


def test_instants_find_the_fastest_rise_before_the_peak():
    time = np.arange(0., 5., .1)
    voltage = np.where(time < 2., -80.+50.*time, 20.-25.*(time-2.))
    at = instants(time, voltage, _row())
    assert at["at_threshold"] == 1. and at["at_peak"] == 2. and at["at_minimum"] == 4.
    assert 0. <= at["at_max_rise"] <= 2.


def test_budgets_integrate_windows_and_read_landmark_currents():
    time = np.arange(0., 5., .1)
    voltage = np.where(time < 2., -80.+50.*time, 20.-25.*(time-2.))
    currents = {"applied": np.full_like(time, .2), "K": np.where(time >= 2., -.5, 0.)}
    entry = budgets(time, voltage, currents, [_row()])[0]
    assert entry["upstroke"]["applied"] == pytest.approx(.2, abs=1e-6)
    assert entry["falling"]["K"] == pytest.approx(-1., abs=1e-6)
    assert entry["at_minimum"]["K"] == pytest.approx(-.5) and entry["at_minimum"]["voltage_mv"] == pytest.approx(-30.)
    assert shares(entry, "upstroke")["applied"] == pytest.approx(1.)


def test_main_reports_only_runs_present_on_disk(tmp_path, monkeypatch):
    root = tmp_path/"a"/"b"/"docs"/"evidence"
    root.mkdir(parents=True)
    folder = root/"h01-x"
    folder.mkdir()
    manifest = {"driver": "h01_pv_neuron_reference.py", "output_dir": "h01-x", "inputs": {"019": []},
                "candidates": [{"name": "r-a", "stage": "R"}, {"name": "r-b", "stage": "R"}]}
    path = root/"m.json"
    path.write_text(json.dumps(manifest))
    time = np.arange(0., 400., .05)
    voltage = np.full_like(time, -80.)+100.*np.exp(-((time-300.)/.5)**2)
    np.savez(folder/"r-a-019.npz", time_ms=time, voltage_mv=voltage, total_current_na=np.full_like(time, .19),
             NaTg_current_ma_cm2=np.zeros_like(time), capacitive_current_ma_cm2=np.zeros_like(time),
             axial_neighbour_0_mv=voltage)
    (folder/"r-a-019.json").write_text(json.dumps({"charge_balance_geometry": {
        "area_um2": 100., "neighbours": [{"voltage_key": "axial_neighbour_0_mv", "resistance_mohm": 10.}]}}))
    main(["--manifest", str(path)])
    report = json.loads((root/"h01-x-stage-r-budgets.json").read_text())
    assert list(report["runs"]) == ["r-a"] and report["runs"]["r-a"]["019"]["spikes"] == 1
