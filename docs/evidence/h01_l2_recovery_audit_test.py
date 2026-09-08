"""Test recovery decisions without relying on a spike-count proxy."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from docs.evidence import h01_l2_recovery_audit as audit


@pytest.fixture
def runs(tmp_path):
    template = json.loads(Path(__file__).with_name("h01-l2-source-sweep50-space3.json").read_text())
    stems = [tmp_path / "control", tmp_path / "candidate"]
    t = np.arange(21001) / 10.
    for index, stem in enumerate(stems):
        meta = json.loads(json.dumps(template))
        if index:
            for row in meta["applied_genome"]:
                if row["name"] == "decay_CaDynamics":
                    row["value"] *= 2
        stem.with_suffix(".json").write_text(json.dumps(meta))
        v = np.full(t.shape, -70.)
        for onset in (1100 + np.arange(4) * (120 if index else 100)):
            v += 90 * np.clip(1 - np.abs(t - onset) / .2, 0, 1)
        np.savez(stem.with_suffix(".npz"), time_ms=t, voltage_mv=v, applied_current_na=np.zeros_like(t))
    human = tmp_path / "human.json"
    human.write_text(json.dumps({"specimen_id": 541563728, "sweep": 50, "intervals_ms": [150., 150., 150.]}))
    return *stems, human


@pytest.mark.parametrize("case,decision", [
    ("improved", "supported"), ("one_interval_worse", "rejected"),
    ("minimum", "rejected"), ("peak", "rejected"),
    ("missing_event", "invalid"), ("nan", "invalid"), ("clipped", "invalid"),
    ("input", "invalid"), ("metadata", "invalid"), ("bias", "invalid"),
    ("wrong_genome", "invalid"), ("missing_target", "invalid"), ("human", "invalid"),
    ("nonspike", "invalid"), ("nan_human", "invalid"),
])
def test_all_conditions_and_invalidity(runs, case, decision):
    control, candidate, human = runs
    with np.load(candidate.with_suffix(".npz")) as data:
        t, v, current = [data[k].copy() for k in ("time_ms", "voltage_mv", "applied_current_na")]
    if case == "one_interval_worse":
        v[:] = -70.
        for onset in [1100, 1220, 1340, 1390]:
            v += 90 * np.clip(1 - np.abs(t - onset) / .2, 0, 1)
    elif case == "minimum":
        v[t == 1150] = -70.11
    elif case == "peak":
        v[t == 1100] += .11
    elif case == "missing_event":
        v[t > 1400] = -70.
    elif case == "nan":
        v[10] = np.nan
    elif case == "clipped":
        v[t == 1020] = 20.
    elif case == "input":
        current[10] = .001
    elif case == "nonspike":
        v[t == 1340] = -1.
    np.savez(candidate.with_suffix(".npz"), time_ms=t, voltage_mv=v, applied_current_na=current)
    path = candidate.with_suffix(".json")
    meta = json.loads(path.read_text())
    if case == "metadata":
        meta["temperature_c"] += 1
    elif case == "bias":
        meta["added_bias_na"] = -.003
    elif case == "wrong_genome":
        meta["applied_genome"][0]["value"] *= 2
    elif case == "missing_target":
        source = json.loads(control.with_suffix(".json").read_text())
        source["applied_genome"] = [r for r in source["applied_genome"] if r["name"] != "decay_CaDynamics"]
        control.with_suffix(".json").write_text(json.dumps(source))
    elif case == "human":
        human.write_text(json.dumps({"specimen_id": 1, "sweep": 50, "intervals_ms": [150.]*3}))
    elif case == "nan_human":
        human.write_text(json.dumps({"specimen_id": 541563728, "sweep": 50, "intervals_ms": [np.nan]*3}))
    path.write_text(json.dumps(meta))
    report = audit.compare_runs(*runs)
    assert report["decision"] == decision
    if case == "one_interval_worse":
        assert report["each_interval_improved"] == [True, True, False]
    assert bool(report["invalid_reasons"]) == (decision == "invalid")


def test_cli_retains_individual_intervals(runs, tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", ["audit", *map(str, runs), str(output)])
    runpy.run_path(audit.__file__, run_name="__main__")
    report = json.loads(output.read_text())
    assert report["decision"] == "supported"
    assert len(report["interval_errors_ms"]) == 2
    assert all(len(row) == 3 for row in report["interval_errors_ms"])
