"""Exercise sodium prediction gates and retained later-event losses."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from docs.evidence import h01_l2_sodium_audit as audit


@pytest.fixture
def runs(tmp_path):
    template = json.loads(Path(__file__).with_name("h01-l2-sodium-recovery-control.json").read_text())
    stems = [tmp_path / "control", tmp_path / "candidate"]
    t = np.arange(21001) / 10.
    for index, stem in enumerate(stems):
        meta = dict(template, sodium_recovery_factor=float(index + 1))
        stem.with_suffix(".json").write_text(json.dumps(meta))
        v = np.full(t.shape, -70.)
        for onset in 1100 + np.arange(4) * (120 if index else 100):
            v += 90 * np.clip(1 - np.abs(t - onset) / .2, 0, 1)
        np.savez(stem.with_suffix(".npz"), time_ms=t, voltage_mv=v, applied_current_na=np.zeros_like(t))
    human = tmp_path / "human.json"
    human.write_text(json.dumps(dict(specimen_id=541563728, sweep=50, intervals_ms=[150.] * 3)))
    return *stems, human


@pytest.mark.parametrize("case,decision", [
    ("improved", "supported"), ("two_events", "supported"),
    ("interval", "rejected"), ("onset", "rejected"), ("peak", "rejected"),
    ("one_event", "invalid"), ("input", "invalid"), ("source", "invalid"),
    ("library", "invalid"), ("factor", "invalid"), ("human", "invalid"),
    ("empty_human", "invalid"),
])
def test_prediction_and_isolation(runs, case, decision):
    _, candidate, human = runs
    with np.load(candidate.with_suffix(".npz")) as data:
        t, v, current = [data[k].copy() for k in ("time_ms", "voltage_mv", "applied_current_na")]
    if case in ("interval", "onset"):
        v[:] = -70.
        for onset in (1100 + (.2 if case == "onset" else 0) +
                      np.arange(4) * (80 if case == "interval" else 120)):
            v += 90 * np.clip(1 - np.abs(t - onset) / .2, 0, 1)
    elif case == "peak":
        v[t == 1100] += .11
    elif case == "two_events":
        v[t > 1300] = -70.
    elif case == "one_event":
        v[t > 1150] = -70.
    elif case == "input":
        current[10] = .01
    np.savez(candidate.with_suffix(".npz"), time_ms=t, voltage_mv=v, applied_current_na=current)
    path = candidate.with_suffix(".json")
    meta = json.loads(path.read_text())
    if case == "source":
        meta["mechanism_source_sha256"] = {}
    elif case == "library":
        meta["mechanism_library_sha256"] = "different"
    elif case == "factor":
        meta["sodium_recovery_factor"] = 3.
    elif case in ("human", "empty_human"):
        reference = json.loads(human.read_text())
        reference["intervals_ms"] = [] if case == "empty_human" else [float("nan")]
        human.write_text(json.dumps(reference))
    path.write_text(json.dumps(meta))
    report = audit.compare_runs(*runs)
    assert report["decision"] == decision
    assert bool(report["invalid_reasons"]) == (decision == "invalid")
    if case == "two_events":
        assert report["event_counts"] == [4, 2]
        assert len(report["records"][1]["events"]) == 2


def test_cli(runs, tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["audit", *map(str, runs), str(output)])
    runpy.run_path(audit.__file__, run_name="__main__")
    report = json.loads(output.read_text())
    assert report["decision"] == "supported"
    assert report["event_counts"] == [4, 4]
