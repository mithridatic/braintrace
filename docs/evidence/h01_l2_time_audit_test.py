"""Test temporal decisions using individual synthetic voltage events."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("key", ["applied_passive_parameters", "leak_reversal_shift_mv"])
def test_changed_passive_settings_are_invalid(runs, key):
    from docs.evidence.h01_l2_time_audit import compare_runs
    for stem, value in zip(runs, (1, 2)):
        path = stem.with_suffix(".json")
        meta = json.loads(path.read_text())
        meta[key] = value
        path.write_text(json.dumps(meta))
    assert compare_runs(*runs)["decision"] == "invalid"

from docs.evidence import h01_l2_time_audit as audit


@pytest.mark.parametrize("key", ["sodium_opening_factor", "kv3_closing_factor"])
@pytest.mark.parametrize("values,decision", [((1, 2), "invalid"),
                         ((None, 1), "invalid"), ((1, None), "invalid"),
                         ((1, 1), "supported")])
def test_opening_factor_invariant(runs, values, decision, key):
    for stem, value in zip(runs, values):
        path = stem.with_suffix(".json")
        meta = json.loads(path.read_text())
        if value is not None:
            meta[key] = value
        path.write_text(json.dumps(meta))
    assert audit.compare_runs(*runs)["decision"] == decision


@pytest.fixture
def runs(tmp_path):
    """Create paired setups with three complete events."""
    meta = json.loads(Path(__file__).with_name("h01-l2-source-sweep50.json").read_text())
    stems = [tmp_path / "coarse", tmp_path / "fine"]
    for index, stem in enumerate(stems):
        meta["dt_ms"] = .005 / (index + 1)
        stem.with_suffix(".json").write_text(json.dumps(meta))
        time = [1000., 1020., 1100., 1100.1, 1100.2, 1200., 1200.1,
                1200.2, 1900., 1900.1, 1900.2, 2019., 2020.]
        voltage = [-70., -70., -70., 20., -70., -70., 20., -70.,
                   -70., 20., -70., -70., -70.]
        np.savez(stem.with_suffix(".npz"), time_ms=time, voltage_mv=voltage)
    return stems


@pytest.mark.parametrize("case,decision", [
    ("same", "supported"), ("onset", "rejected"), ("peak", "rejected"),
    ("width", "rejected"), ("count", "rejected"), ("sign", "rejected"),
    ("empty", "invalid"), ("nan", "invalid"), ("clock", "invalid"),
    ("coverage", "invalid"), ("clipped", "invalid"), ("shape", "invalid"),
    ("metadata", "invalid"), ("missing_metadata", "invalid"), ("step", "invalid"),
    ("nan_step", "invalid"),
    ("bias", "invalid"),
    ("missing_bias", "invalid"),
])
def test_direct_response_decision(runs, case, decision):
    """Keep failures distinct from missing evidence and changed setups."""
    path = runs[1].with_suffix(".npz")
    with np.load(path) as data:
        t, v = data["time_ms"].copy(), data["voltage_mv"].copy()
    if case == "onset":
        t[5:8] += .11
    elif case == "peak":
        v[6] += .2
    elif case == "width":
        t[7] += .1
    elif case == "count":
        v[6] = -70
    elif case == "sign":
        v[6] = -1
    elif case == "empty":
        v[:] = -70
    elif case == "nan":
        v[5] = np.nan
    elif case == "clock":
        t[5] = t[4]
    elif case == "coverage":
        t[-1] = 2019.5
    elif case == "clipped":
        v[-2] = 20
    elif case == "shape":
        v = v[:-1]
    np.savez(path, time_ms=t, voltage_mv=v)
    meta_path = runs[1].with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    if case == "metadata":
        meta["temperature_c"] += 1
    elif case == "missing_metadata":
        del meta["sections"]
    elif case == "step":
        meta["dt_ms"] = .005
    elif case == "nan_step":
        meta["dt_ms"] = float("nan")
    elif case == "bias":
        meta["added_bias_na"] = -.003
    elif case == "missing_bias":
        meta["input"] = "command plus recorded bias"
    meta_path.write_text(json.dumps(meta))
    report = audit.compare_runs(*runs)
    assert report["decision"] == decision
    assert bool(report["invalid_reasons"]) == (decision == "invalid")
    if case == "count":
        assert report["counts"] == [3, 2]
        assert len(report["unmatched_events"]["coarse"]) == 1


def test_command_line_writes_complete_report(runs, monkeypatch, tmp_path):
    """Exercise the same entry point used for recorded traces."""
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["audit", *map(str, runs), str(output)])
    runpy.run_path(audit.__file__, run_name="__main__")
    report = json.loads(output.read_text())
    assert report["decision"] == "supported"
    assert len(report["ordinal_differences_fine_minus_coarse"]) == 3
