"""Mesh copy, protocol collapse, and report provenance of the L2 BrainCell runner without a long run."""
import json
from pathlib import Path
from types import SimpleNamespace

import braincell
import brainunit as u
import numpy as np
import pytest

from braintrace.datasets import h01_l2_cell
from braintrace.datasets.h01_pv_morphology import make_pv_morphology
from docs.evidence import h01_l2_braincell_reference as driver

FOLDER = Path(__file__).parent
REFERENCE = json.loads((FOLDER/"h01-l2-geometry-reference.json").read_text())


def test_section_names_map_to_branch_names_once():
    sections = [{"name": "soma[0]", "nseg": 9}, {"name": "apic[12]", "nseg": 27}]
    assert driver.neuron_counts_by_branch(sections) == {"soma_0": 9, "apic_12": 27}
    with pytest.raises(ValueError, match="Duplicate"):
        driver.neuron_counts_by_branch(sections+[sections[0]])


def test_step_protocol_collapses_to_segments_covering_the_stop():
    t = np.arange(0., 3000., .02)
    c = np.where((t >= 5.) & (t < 15.02), .05, 0.)+np.where((t >= 1020.) & (t < 2020.), .31, 0.)
    durations, amplitudes = driver.constant_segments(t, c, 2100.)
    assert amplitudes == [0., .05, 0., .31, 0.]
    assert np.allclose(durations, [5., 10.02, 1004.98, 1000., 80.]) and sum(durations) == pytest.approx(2100.)
    durations, amplitudes = driver.constant_segments(t, c, 1020.)
    assert amplitudes == [0., .05, 0.] and sum(durations) == pytest.approx(1020.)


@pytest.mark.parametrize("bad", [dict(stop=5000.), dict(stop=0.), dict(start=1.), dict(nan=True)])
def test_invalid_protocol_is_rejected(bad):
    t = np.arange(0., 100., .02)+bad.get("start", 0.)
    c = np.zeros_like(t)
    if bad.get("nan"):
        c[3] = np.nan
    with pytest.raises(ValueError):
        driver.constant_segments(t, c, bad.get("stop", 50.))


def _cache(tmp_path, bias=-.0037):
    t = np.arange(0., 3000., .02)
    c = np.where((t >= 1020.) & (t < 2020.), .31, 0.)
    np.savez(tmp_path/"sweep-53.npz", time_ms=t, command_current_na=c, bias_current_na=np.float64(bias))
    return tmp_path


def _spy(monkeypatch, calls):
    def build(reference, **kwargs):
        calls.update(kwargs)
        def run(**settings):
            calls.update(settings)
            return SimpleNamespace(traces={"voltage": np.array([-84., -83.])*u.mV, "calcium": np.array([1e-4, 1e-4])*u.mM})
        return SimpleNamespace(run=run)
    monkeypatch.setattr(h01_l2_cell, "make_l2_cell", build)


def test_mesh_from_l2_report_copies_counts_in_branch_order_and_adds_bias(tmp_path, monkeypatch):
    sections = [{"name": s["name"], "nseg": 2*i+1} for i, s in enumerate(REFERENCE["sections"])]
    report_json = tmp_path/"b3.json"
    report_json.write_text(json.dumps({"sections": sections}))
    calls = {}
    _spy(monkeypatch, calls)
    out = tmp_path/"trace"
    driver.main(["--cache", str(_cache(tmp_path)), "--sweep", "53", "--dt-ms", "1", "--stop-ms", "2100",
                 "--mesh-from", str(report_json), "--output", str(out)])
    branches = [b.name for b in make_pv_morphology(REFERENCE).branches]
    expected = {s["name"].replace("[", "_").replace("]", ""): 2*i+1 for i, s in enumerate(REFERENCE["sections"])}
    assert isinstance(calls["cv_policy"], braincell.CVPerBranchList)
    assert list(calls["cv_policy"].cv_per_branch) == [expected[b] for b in branches]
    assert calls["current_na"] == pytest.approx([-.0037, .31-.0037, -.0037]) and sum(calls["duration_ms"]) == 2100.
    assert calls["mode"] == "candidate" and float(calls["duration"].to_decimal(u.ms)) == 2100.
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["mesh"]["policy"] == "CVPerBranchList" and report["mesh"]["branches"] == branches
    assert report["mesh"]["source"] == "b3.json" and report["sweep"] == 53 and report["added_bias_na"] == -.0037
    assert report["segments"]["amplitudes_na"] == calls["current_na"] and report["spike_times_ms"] == []


def test_default_mesh_and_command_only_input(tmp_path, monkeypatch):
    calls = {}
    _spy(monkeypatch, calls)
    out = tmp_path/"trace"
    driver.main(["--cache", str(_cache(tmp_path)), "--sweep", "53", "--dt-ms", "1", "--max-cv-um", "2.5", "--command-only",
                 "--profile-key", "source", "--output", str(out)])
    assert isinstance(calls["cv_policy"], type(None)) and calls["max_cv_length_um"] == 2.5
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["added_bias_na"] == 0. and report["input"].startswith("source command only")
    assert report["profile"]["mode"] == "source" and report["max_cv_length_um"] == 2.5


def test_incomplete_report_and_invalid_window_stop_before_the_model(tmp_path):
    report_json = tmp_path/"b3.json"
    report_json.write_text(json.dumps({"sections": [{"name": "soma[0]", "nseg": 9}]}))
    with pytest.raises(SystemExit) as error:
        driver.main(["--cache", str(tmp_path), "--mesh-from", str(report_json), "--output", str(tmp_path/"x")])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        driver.main(["--cache", str(tmp_path), "--dt-ms", "0", "--output", str(tmp_path/"x")])
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--profile-key", "--mode"])
def test_b3_mode_reaches_the_builder_and_the_report(tmp_path, monkeypatch, flag):
    calls = {}
    _spy(monkeypatch, calls)
    out = tmp_path/"trace"
    driver.main(["--cache", str(_cache(tmp_path)), "--sweep", "53", "--dt-ms", "1", flag, "b3", "--output", str(out)])
    report = json.loads(out.with_suffix(".json").read_text())
    assert calls["mode"] == "b3" and report["profile_key"] == "b3" and report["profile"]["mode"] == "b3"
    assert report["profile"]["regions"][1][3] == [["NaTs", 3.814]]
    assert report["initial_voltage_mv"] == -83.97993469238281
