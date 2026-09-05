"""Driver provenance and short-window behavior without a long simulation."""
import json
import runpy
import sys
from types import SimpleNamespace
import numpy as np
import brainunit as u
import pytest
from braintrace.datasets import h01_pv_cell


@pytest.mark.parametrize("mode", ["candidate", "source"])
def test_driver_records_selected_profile_and_requested_window(tmp_path, monkeypatch, mode):
    calls = {}
    def build(reference, current, **kwargs):
        calls.update(kwargs)
        def run(**settings):
            calls.update(settings)
            return SimpleNamespace(traces={"voltage": np.array([-80., -79.])*u.mV,
                "calcium": np.array([1e-4, 1e-4])*u.mM, "sk_gate": np.array([.1, .1])})
        return SimpleNamespace(run=run)
    monkeypatch.setattr(h01_pv_cell, "make_pv_cell", build)
    output = tmp_path/"trace"
    monkeypatch.setattr(sys, "argv", ["driver", "--mode", mode, "--duration-ms", "2",
        "--dt-ms", "1", "--output", str(output)])
    runpy.run_module("docs.evidence.h01_pv_braincell_reference", run_name="__main__")
    report = json.loads(output.with_suffix(".json").read_text())
    assert calls["mode"] == mode
    assert float(calls["duration"].to_decimal(u.ms)) == 2.
    assert report["profile"]["mode"] == mode
    assert report["profile"]["metadata_sha256"]
    assert report["baseline_mean_mv"] is None
    assert report["duration_ms"] == 2.


@pytest.mark.parametrize("flag,value", [("--duration-ms", "0"), ("--duration-ms", "nan"), ("--dt-ms", "-1")])
def test_driver_rejects_invalid_window_before_model_build(tmp_path, monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["driver", flag, value, "--output", str(tmp_path/"bad")])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("docs.evidence.h01_pv_braincell_reference", run_name="__main__")
    assert error.value.code == 2


def test_mesh_from_neuron_json_copies_counts_in_branch_order(tmp_path, monkeypatch):
    import braincell
    from braintrace.datasets.h01_pv_morphology import make_pv_morphology
    from pathlib import Path
    reference = json.loads(Path(__file__).with_name("h01-pv-geometry-reference.json").read_text())
    geometry = [{"name": "NeuronTemplate[0]."+s["name"], "nseg": 2*i+1} for i, s in enumerate(reference["sections"])]
    neuron_json = tmp_path/"neuron.json"
    neuron_json.write_text(json.dumps({"geometry": geometry}))
    calls = {}
    def build(reference, current, **kwargs):
        calls.update(kwargs)
        def run(**settings):
            return SimpleNamespace(traces={"voltage": np.array([-80., -79.])*u.mV,
                "calcium": np.array([1e-4, 1e-4])*u.mM, "sk_gate": np.array([.1, .1])})
        return SimpleNamespace(run=run)
    monkeypatch.setattr(h01_pv_cell, "make_pv_cell", build)
    output = tmp_path/"trace"
    monkeypatch.setattr(sys, "argv", ["driver", "--duration-ms", "2", "--dt-ms", "1",
        "--mesh-from", str(neuron_json), "--output", str(output)])
    runpy.run_module("docs.evidence.h01_pv_braincell_reference", run_name="__main__")
    expected = {s["name"].replace("[", "_").replace("]", ""): 2*i+1 for i, s in enumerate(reference["sections"])}
    branches = [b.name for b in make_pv_morphology(reference).branches]
    assert isinstance(calls["cv_policy"], braincell.CVPerBranchList)
    assert list(calls["cv_policy"].cv_per_branch) == [expected[b] for b in branches]
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["mesh"]["policy"] == "CVPerBranchList" and report["max_cv_length_um"] is None
    assert report["mesh"]["branches"] == branches and report["mesh"]["source"] == "neuron.json"


def test_mesh_from_incomplete_geometry_is_rejected(tmp_path, monkeypatch):
    neuron_json = tmp_path/"neuron.json"
    neuron_json.write_text(json.dumps({"geometry": [{"name": "NeuronTemplate[0].soma[0]", "nseg": 9}]}))
    monkeypatch.setattr(sys, "argv", ["driver", "--mesh-from", str(neuron_json), "--output", str(tmp_path/"bad")])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("docs.evidence.h01_pv_braincell_reference", run_name="__main__")
    assert error.value.code == 2
