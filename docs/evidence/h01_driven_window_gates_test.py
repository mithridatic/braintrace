"""Tests for the driven-window gate driver on small synthetic populations."""

import json

import numpy as np
import pytest

from docs.evidence import h01_driven_window_gates as gates


def _write(tmp_path, name, obj):
    path = tmp_path/name
    path.write_text(json.dumps(obj))
    return path


def test_decide_reports_fail_and_hashes_when_runtime_gate_rejects(tmp_path, monkeypatch):
    reference = _write(tmp_path, "reference.json", {"simulated_cell_ids": ["a"], "contacts": []})
    plan = _write(tmp_path, "plan.json", {"cells": 1, "control": "ei", "dt_ms": .5, "duration_ms": 1.})
    folder = tmp_path/"run"
    folder.mkdir()
    (folder/"run-build.json").write_text(json.dumps({"control": "ei"}))
    np.savez(folder/"run-traces.npz", time_ms=np.array([.5, 1.]))
    runs = {control: [str(folder), str(plan)] for control in gates.CONTROLS}
    decision = gates.decide(str(reference), runs, {"coarse": [str(folder), str(plan)], "fine": [str(folder), str(plan)]})
    assert decision["verdict"] == "FAIL" and decision["driven_window"] == 0.
    assert all(v["status"] == "failed" for v in decision["runtime"].values())
    assert decision["controls"]["status"] == "failed"
    assert decision["refinement"]["status"] == "failed"
    assert set(decision["input_hashes"]) == {"reference", *gates.CONTROLS, "refinement_coarse", "refinement_fine"}
    assert all(len(h["build"]) == 64 for c, h in decision["input_hashes"].items() if c in gates.CONTROLS)


def test_decide_passes_only_when_every_gate_passes(tmp_path, monkeypatch):
    passed = dict(status="passed", failures=[])
    monkeypatch.setattr(gates, "audit_runtime", lambda *a: passed)
    monkeypatch.setattr(gates, "audit_controls", lambda *a: passed)
    monkeypatch.setattr(gates, "audit_refinement", lambda *a: dict(
        status="passed", failures=[], observations=[dict(voltage_errors_mV=dict(v=.2), max_event_timing_error_ms=None)]))
    reference = _write(tmp_path, "reference.json", {"simulated_cell_ids": ["a"], "contacts": []})
    plan = _write(tmp_path, "plan.json", {"cells": 1})
    folder = tmp_path/"run"
    folder.mkdir()
    (folder/"run-build.json").write_text("{}")
    np.savez(folder/"run-traces.npz", time_ms=np.array([1.]))
    runs = {control: [str(folder), str(plan)] for control in gates.CONTROLS}
    decision = gates.decide(str(reference), runs, {"coarse": [str(folder), str(plan)], "fine": [str(folder), str(plan)]})
    assert decision["verdict"] == "PASS" and decision["driven_window"] == 1.
    assert decision["refinement"]["worst_voltage_mV"] == pytest.approx(.2)
    assert decision["refinement"]["worst_event_ms"] == 0.
    del runs["disconnected"]
    partial = gates.decide(str(reference), runs, {"coarse": [str(folder), str(plan)], "fine": [str(folder), str(plan)]})
    assert partial["controls"] is None and partial["verdict"] == "FAIL"
