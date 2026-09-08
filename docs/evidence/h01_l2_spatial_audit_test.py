"""Check structural failures before interpreting spatial event differences."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from docs.evidence import h01_l2_spatial_audit as audit
from docs.evidence.h01_l2_time_audit_test import runs


@pytest.mark.parametrize("key", ["sodium_opening_factor", "kv3_closing_factor"])
@pytest.mark.parametrize("values,decision", [((1, 2), "invalid"),
                         ((None, 1), "invalid"), ((1, None), "invalid"),
                         ((1, 1), "supported")])
def test_opening_factor_invariant(spatial_runs, values, decision, key):
    for stem, value in zip(spatial_runs[:2], values):
        path = stem.with_suffix(".json")
        meta = json.loads(path.read_text())
        if value is not None:
            meta[key] = value
        path.write_text(json.dumps(meta))
    assert audit.compare_runs(*spatial_runs)["decision"] == decision


@pytest.mark.parametrize("key", ["sodium_recovery_factor", "mechanism_source_sha256",
                                  "mechanism_library_sha256", "calcium_removal_intervention",
                                  "applied_passive_parameters", "leak_reversal_shift_mv"])
def test_changed_intervention_metadata_is_invalid(spatial_runs, key):
    coarse, fine, topology = spatial_runs
    for stem, value in ((coarse, 1), (fine, 2)):
        path = stem.with_suffix(".json")
        meta = json.loads(path.read_text())
        meta[key] = value
        path.write_text(json.dumps(meta))
    assert audit.compare_runs(coarse, fine, topology)["decision"] == "invalid"


@pytest.fixture
def spatial_runs(runs, tmp_path):
    topology = Path(__file__).with_name("h01-l2-source-topology-audit.json")
    replay = json.loads(topology.read_text())
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps(replay))
    meta = json.loads(runs[1].with_suffix(".json").read_text())
    meta["dt_ms"] = .005
    meta["nseg_factor"] = 3
    for section in meta["sections"]:
        section["nseg"] *= 3
        section["parent"] = replay["parent_sections"][section["name"]]
    runs[1].with_suffix(".json").write_text(json.dumps(meta))
    return *runs, topology


@pytest.mark.parametrize("case,decision", [
    ("same", "supported"), ("onset", "rejected"), ("empty", "invalid"),
    ("parent", "invalid"), ("missing_parent", "invalid"), ("segments", "invalid"),
    ("area", "invalid"), ("capacitance", "invalid"), ("bias", "invalid"),
    ("step", "invalid"), ("factor", "invalid"), ("section", "invalid"),
    ("topology", "invalid"), ("explicit_parent", "supported"),
])
def test_structural_and_direct_event_decisions(spatial_runs, case, decision):
    coarse, fine, topology = spatial_runs
    path = fine.with_suffix(".json")
    meta = json.loads(path.read_text())
    if case == "parent":
        meta["sections"][0]["parent"] = "dend[0]"
    elif case == "missing_parent":
        del meta["sections"][0]["parent"]
    elif case == "segments":
        meta["sections"][-1]["nseg"] //= 3
    elif case == "area":
        meta["sections"][0]["area_um2"] *= 1.01
    elif case == "capacitance":
        meta["sections"][0]["cm_uf_cm2"] *= 2
    elif case == "bias":
        meta["added_bias_na"] = -.003
    elif case == "step":
        meta["dt_ms"] /= 2
    elif case == "factor":
        meta["nseg_factor"] = 9
    elif case == "section":
        meta["sections"].pop()
    elif case == "topology":
        replay = json.loads(topology.read_text())
        replay["all_sections_reach_soma"] = False
        topology.write_text(json.dumps(replay))
    elif case == "explicit_parent":
        original = json.loads(coarse.with_suffix(".json").read_text())
        replay = json.loads(topology.read_text())
        for section in original["sections"]:
            section["parent"] = replay["parent_sections"][section["name"]]
        coarse.with_suffix(".json").write_text(json.dumps(original))
    path.write_text(json.dumps(meta))
    if case in ("onset", "empty"):
        with np.load(fine.with_suffix(".npz")) as data:
            t, v = data["time_ms"].copy(), data["voltage_mv"].copy()
        if case == "onset":
            t[5:8] += .11
        else:
            v[:] = -70
        np.savez(fine.with_suffix(".npz"), time_ms=t, voltage_mv=v)
    report = audit.compare_runs(*spatial_runs)
    assert report["decision"] == decision
    if case == "same":
        assert report["segment_counts"][1] == 3 * report["segment_counts"][0]
        assert len(report["sections_using_replayed_parent"]) == 91
    elif case == "explicit_parent":
        assert report["sections_using_replayed_parent"] == []


def test_cli_preserves_replay_limit(spatial_runs, tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["audit", *map(str, spatial_runs), str(output)])
    runpy.run_path(audit.__file__, run_name="__main__")
    report = json.loads(output.read_text())
    assert report["decision"] == "supported"
    assert "not original-run" in report["baseline_parent_evidence"]


def test_successive_factor_three_to_nine(spatial_runs):
    for index, stem in enumerate(spatial_runs[:2]):
        path = stem.with_suffix(".json")
        meta = json.loads(path.read_text())
        meta["nseg_factor"] = 3 if index == 0 else 9
        for section in meta["sections"]:
            section["nseg"] *= 3
        path.write_text(json.dumps(meta))
    report = audit.compare_runs(*spatial_runs)
    assert report["decision"] == "supported"
