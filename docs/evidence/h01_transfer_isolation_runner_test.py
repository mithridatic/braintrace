"""Manifest checks, exact commands, alignment, per-event scoring, and the decision rule on fixtures."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from braintrace.datasets.h01_ei_profiles import channel_controls, get_ei_profile
from docs.evidence import h01_transfer_isolation_runner as runner

FOLDER = Path(__file__).parent
MANIFEST = json.loads((FOLDER/"h01-transfer-i-manifest.json").read_text())
ROOT = FOLDER.parents[1]


def test_registered_manifest_passes_checks_and_names_every_ordered_arm():
    names = runner.check_manifest(MANIFEST)
    assert sorted(names) == sorted(MANIFEST["order"])
    assert (FOLDER/MANIFEST["output_dir"]/"b1-fixed.candidate.json").exists()
    assert (FOLDER/MANIFEST["mesh_source"]).exists()
    for stem in MANIFEST["references"].values():
        assert (FOLDER/MANIFEST["reference_dir"]/(stem+".json")).exists()


def test_b1_candidate_is_the_finalist_without_cvode():
    finalist = json.loads((FOLDER/MANIFEST["finalist_candidate_json"]).read_text())
    b1 = json.loads((FOLDER/MANIFEST["output_dir"]/"b1-fixed.candidate.json").read_text())
    assert "cvode_atol" not in b1
    for key, value in b1.items():
        assert finalist[key] == value
    assert set(finalist)-set(b1) == {"cvode_atol", "observe_charge_balance", "observe_spike_currents"}


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["arms"].append(dict(m["arms"][0])), "unique"),
    (lambda m: m["arms"][0].pop("dt_ms"), "lacks dt_ms"),
    (lambda m: m["arms"][0].update(simulator="hoc"), "unknown simulator"),
    (lambda m: m["arms"][0].update(reference="031"), "unknown reference"),
    (lambda m: m["arms"][3].pop("max_cv_um"), "without max_cv_um"),
    (lambda m: m["arms"][7].pop("candidate_json"), "needs a candidate_json"),
    (lambda m: m["arms"][1].update(dt_ms=.005), "half the dt"),
    (lambda m: m["caps"].update(neuron=1), "exceed the cap"),
    (lambda m: m.pop("rejection"), "pre-registered rejection"),
])
def test_manifest_violations_are_rejected(mutate, match):
    manifest = copy.deepcopy(MANIFEST)
    mutate(manifest)
    with pytest.raises(ValueError, match=match):
        runner.check_manifest(manifest)


def test_braincell_commands_copy_the_finalist_mesh_or_the_current_max_cv():
    arms = {arm["name"]: arm for arm in MANIFEST["arms"]}
    a1 = runner.arm_command(ROOT, MANIFEST, arms["a1-matched-019"])
    assert a1[1:3] == ["-m", "docs.evidence.h01_pv_braincell_reference"]
    assert a1[a1.index("--mesh-from")+1].endswith("e-kv3-close2-027.json")
    assert a1[a1.index("--current-na")+1] == "0.19" and a1[a1.index("--dt-ms")+1] == "0.005"
    assert a1[a1.index("--duration-ms")+1] == "1500.0" and a1[-1].endswith("a1-matched-019")
    a0 = runner.arm_command(ROOT, MANIFEST, arms["a0-maxcv-027"])
    assert a0[a0.index("--max-cv-um")+1] == "2.5" and "--mesh-from" not in a0


def test_neuron_commands_are_fixed_step_in_the_pinned_container():
    arms = {arm["name"]: arm for arm in MANIFEST["arms"]}
    b1 = runner.arm_command(ROOT, MANIFEST, arms["b1-fixed-027"])
    assert b1[:3] == ["docker", "run", "--rm"] and "braintrace-h01-neuron:9.0.2" in b1
    assert "--cvode-atol" not in b1 and b1[b1.index("--dt-ms")+1] == "0.005"
    assert b1[b1.index("--candidate-json")+1] == "/evidence/h01-i-transfer/b1-fixed.candidate.json"
    assert b1[-1] == "/evidence/h01-i-transfer/b1-fixed-027"
    half = runner.arm_command(ROOT, MANIFEST, arms["r-neuron-fixed-027-halfdt"])
    assert half[half.index("--dt-ms")+1] == "0.0025" and half[half.index("--duration-ms")+1] == "330.0"


def test_current_candidate_profile_is_not_aligned_with_the_finalist():
    meta = json.loads((FOLDER/MANIFEST["mesh_source"]).read_text())
    report = runner.profile_alignment(meta, get_ei_profile("I"), channel_controls)
    flags = {row["flag"] for row in report["mismatches"]}
    assert not report["aligned"] and flags == {"somatic_kv3_close_factor"}
    assert report["mismatches"][0]["neuron"] == 2. and report["mismatches"][0]["braincell"] == .5


def test_aligned_profile_reports_no_mismatch():
    meta = json.loads((FOLDER/MANIFEST["mesh_source"]).read_text())
    meta = {**meta, "somatic_kv3_close_factor": .5}
    assert runner.profile_alignment(meta, get_ei_profile("I"), channel_controls)["aligned"]
    wrong = {**meta, "axon_calcium_gamma": .005, "conductance_intervention": {"mechanism": "SK", "region": "all"}}
    flags = {row["flag"] for row in runner.profile_alignment(wrong, get_ei_profile("I"), channel_controls)["mismatches"]}
    assert flags == {"axon_calcium_gamma", "conductance_intervention"}


def _trace(shift=0., events=8, drop=0):
    t = np.arange(0., 340.001, .01)
    v = np.full_like(t, -80.)
    for k in range(events-drop):
        v += 120.*np.exp(-((t-(275.+6.*k+(shift if k >= 6 else 0.)))/.3)**2)
    return {"time_ms": t, "voltage_mv": v}


def test_score_pair_reports_rows_and_the_first_failed_event():
    gate = MANIFEST["gate"]
    same = runner.score_pair(_trace(), _trace(), (270., 329.5), gate)
    assert same["passed"] and same["first_failed_event"] is None and len(same["events"]) == 8
    drift = runner.score_pair(_trace(), _trace(shift=-.3), (270., 329.5), gate)
    assert not drift["passed"] and drift["first_failed_event"] == 7
    assert drift["max_abs_rise_error_ms"] == pytest.approx(.3, abs=1e-6)
    fewer = runner.score_pair(_trace(), _trace(drop=1), (270., 329.5), gate)
    assert not fewer["passed"] and fewer["first_failed_event"] == "count" and fewer["actual_count"] == 7


@pytest.mark.parametrize("a1,b1,valid,expected", [
    ({"x": True, "y": True}, {"x": True, "y": True}, True, "mesh"),
    ({"x": True, "y": False}, {"x": True, "y": True}, True, "implementation_difference_at_event_31"),
    ({"x": False, "y": False}, {"x": True, "y": True}, True, "implementation_difference_at_event_count"),
    ({"x": False, "y": True}, {"x": False, "y": True}, True, "integration"),
    ({"x": True, "y": True}, {"x": False, "y": True}, True, "integration"),
    ({"x": True, "y": True}, {"x": True, "y": True}, False, "time_level_open"),
])
def test_decision_rule(a1, b1, valid, expected):
    failures = {"x": None, "y": 31} if expected.endswith("31") else {"x": "count", "y": "count"}
    assert runner.decide(a1, b1, valid, failures) == expected


def test_score_manifest_marks_absent_traces_untested_and_scores_present_ones(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    root = tmp_path
    out = root/"docs/evidence"/manifest["output_dir"]
    ref = root/"docs/evidence"/manifest["reference_dir"]
    out.mkdir(parents=True), ref.mkdir(parents=True)
    for stem in manifest["references"].values():
        np.savez(ref/(stem+".npz"), **_trace())
    for name, shift in (("r-braincell-matched-027", 0.), ("r-braincell-matched-027-halfdt", .01),
                        ("a1-matched-027", 0.), ("a1-matched-019", -.3), ("b1-fixed-027", 0.), ("b1-fixed-019", 0.)):
        np.savez(out/(name+".npz"), **_trace(shift))
    for arm in manifest["arms"]:
        arm["window"] = "halving"
    scored = runner.score_manifest(root, manifest)
    assert scored["scores"]["a0-maxcv-027"]["status"] == "untested"
    assert scored["halving"]["braincell_halving"]["passed"] and scored["halving"]["neuron_halving"]["status"] == "untested"
    report = runner.decision_report(manifest, scored)
    assert report["decision"] == "time_level_open" and report["a1_passed"] == {"a1-matched-019": False, "a1-matched-027": True}
    np.savez(out/"r-neuron-fixed-027-halfdt.npz", **_trace(.02))
    report = runner.decision_report(manifest, runner.score_manifest(root, manifest))
    assert report["gate_valid"] and report["decision"] == "implementation_difference_at_event_7"


def test_main_prints_commands_and_refuses_to_score_an_unaligned_profile(capsys):
    runner.main(["--manifest", str(FOLDER/"h01-transfer-i-manifest.json"), "--print-commands"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(MANIFEST["order"]) and lines[0].startswith("r-braincell-matched-027:")
    with pytest.raises(SystemExit, match="not the NEURON finalist"):
        runner.main(["--manifest", str(FOLDER/"h01-transfer-i-manifest.json"), "--score"])
