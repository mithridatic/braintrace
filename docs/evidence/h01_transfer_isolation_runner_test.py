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


def test_main_prints_commands_and_refuses_to_score_an_unaligned_profile(capsys, tmp_path):
    runner.main(["--manifest", str(FOLDER/"h01-transfer-i-manifest.json"), "--print-commands"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(MANIFEST["order"]) and lines[0].startswith("r-braincell-matched-027:")
    unaligned = tmp_path/"candidate-manifest.json"
    unaligned.write_text(json.dumps({**MANIFEST, "braincell_profile_key": "candidate"}))
    with pytest.raises(SystemExit, match="not the NEURON finalist"):
        runner.main(["--manifest", str(unaligned), "--score"])


def test_manifest_names_the_finalist_mode_and_it_is_aligned(capsys):
    assert MANIFEST["braincell_profile_key"] == "finalist" and MANIFEST["polarity"] == "I"
    meta = json.loads((FOLDER/MANIFEST["mesh_source"]).read_text())
    report = runner.profile_alignment(meta, get_ei_profile("I", mode="finalist"), channel_controls)
    assert report["aligned"] and report["mismatches"] == []
    arms = {arm["name"]: arm for arm in MANIFEST["arms"]}
    a1 = runner.arm_command(ROOT, MANIFEST, arms["a1-matched-027"])
    assert a1[a1.index("--mode")+1] == "finalist"
    runner.main(["--manifest", str(FOLDER/"h01-transfer-i-manifest.json"), "--check-alignment"])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["profile_alignment"]["aligned"] is True


def test_peak_method_is_threaded_into_scores_and_the_decision_json(tmp_path, capsys):
    manifest = copy.deepcopy(MANIFEST)
    root = tmp_path
    out = root/"docs/evidence"/manifest["output_dir"]
    ref = root/"docs/evidence"/manifest["reference_dir"]
    out.mkdir(parents=True), ref.mkdir(parents=True)
    for stem in manifest["references"].values():
        np.savez(ref/(stem+".npz"), **_trace())
    np.savez(out/"a1-matched-027.npz", **_trace())
    for arm in manifest["arms"]:
        arm["window"] = "halving"
    scored = runner.score_manifest(root, manifest, peak_method="interpolated")
    row = scored["scores"]["a1-matched-027"]
    assert row["peak_method"] == "interpolated" and set(row["gate"]) == {
        "rise_crossing_ms", "peak_interpolated_voltage_mv", "time_above_threshold_ms"}
    assert {"peak_sample_voltage_mv", "peak_interpolated_voltage_mv"} <= set(row["events"][0]["errors"])
    assert set(row["events"][0]["in_gate"]) == set(row["gate"])
    assert scored["halving"]["braincell_halving"]["status"] == "untested"
    report = runner.decision_report(manifest, scored)
    assert report["peak_method"] == "interpolated" and report["decision"] == "untested"
    default = runner.score_manifest(root, manifest)
    assert default["scores"]["a1-matched-027"]["peak_method"] == "sample"
    assert "peak_sample_voltage_mv" in default["scores"]["a1-matched-027"]["gate"]


def test_reference_root_reads_reference_traces_from_another_tree(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    root, elsewhere = tmp_path/"here", tmp_path/"sibling"
    out = root/"docs/evidence"/manifest["output_dir"]
    ref = elsewhere/"docs/evidence"/manifest["reference_dir"]
    out.mkdir(parents=True), ref.mkdir(parents=True)
    for stem in manifest["references"].values():
        np.savez(ref/(stem+".npz"), **_trace())
    np.savez(out/"a1-matched-027.npz", **_trace())
    for arm in manifest["arms"]:
        arm["window"] = "halving"
    scored = runner.score_manifest(root, manifest, reference_root=elsewhere)
    assert scored["scores"]["a1-matched-027"]["passed"]
    assert scored["reference_sha256"]["027"] and scored["reference_root"] == str(elsewhere)


def test_gate_amendment_names_the_half_millivolt_peak_row_and_the_richardson_pairs():
    assert MANIFEST["gate"] == {"rise_crossing_ms": .1, "peak_sample_voltage_mv": .5, "time_above_threshold_ms": .01}
    assert "0.5 mV" in MANIFEST["gate_amendment"]
    arms = {arm["name"]: arm for arm in MANIFEST["arms"]}
    for name, partner in MANIFEST["richardson_pairs"].items():
        assert arms[partner]["dt_ms"] == pytest.approx(arms[name]["dt_ms"]/2.)
        assert arms[partner]["current_na"] == arms[name]["current_na"]
        assert arms[partner]["simulator"] == arms[name]["simulator"]


def _peak_scaled(scale, events=3):
    t = np.arange(0., 340.001, .01)
    v = np.full_like(t, -80.)
    for k in range(events):
        v += scale*120.*np.exp(-((t-(275.+6.*k))/.3)**2)
    return {"time_ms": t, "voltage_mv": v}


def test_richardson_peaks_extrapolate_first_order_toward_the_reference():
    reference, coarse, fine = _peak_scaled(1.), _peak_scaled(.99), _peak_scaled(.995)
    rows = runner.richardson_peaks(reference, coarse, fine, (270., 329.5), "interpolated")
    assert [row["event"] for row in rows] == [1, 2, 3]
    for row in rows:
        assert row["peak_half_dt_mv"] > row["peak_dt_mv"]
        assert row["peak_richardson_mv"] == pytest.approx(2.*row["peak_half_dt_mv"]-row["peak_dt_mv"])
        assert abs(row["peak_richardson_error_mv"]) < .02
    assert runner.richardson_peaks(reference, coarse, _peak_scaled(.995, events=2), (270., 329.5)) == []


def test_attach_richardson_fills_partnered_events_and_nulls_the_rest():
    score = runner.score_pair(_trace(), _trace(), (270., 329.5), MANIFEST["gate"])
    richardson = [{"event": 1, "peak_richardson_mv": 40.1, "peak_richardson_error_mv": .1}]
    runner.attach_richardson(score, richardson)
    assert score["richardson_events"] == 1
    assert score["events"][0]["peak_richardson_mv"] == 40.1 and score["events"][0]["peak_richardson_error_mv"] == .1
    assert all(row["peak_richardson_mv"] is None and row["peak_richardson_error_mv"] is None
               for row in score["events"][1:])


def test_score_manifest_adds_the_richardson_column_from_the_half_dt_partner(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    root = tmp_path
    out = root/"docs/evidence"/manifest["output_dir"]
    ref = root/"docs/evidence"/manifest["reference_dir"]
    out.mkdir(parents=True), ref.mkdir(parents=True)
    for stem in manifest["references"].values():
        np.savez(ref/(stem+".npz"), **_peak_scaled(1.))
    np.savez(out/"a1-matched-027.npz", **_peak_scaled(.99))
    np.savez(out/"r-braincell-matched-027.npz", **_peak_scaled(.99))
    np.savez(out/"r-braincell-matched-027-halfdt.npz", **_peak_scaled(.995))
    for arm in manifest["arms"]:
        arm["window"] = "halving"
    scored = runner.score_manifest(root, manifest, peak_method="interpolated")
    for name in ("a1-matched-027", "r-braincell-matched-027"):
        rows = scored["scores"][name]["events"]
        assert scored["scores"][name]["richardson_events"] == 3
        assert all(abs(row["peak_richardson_error_mv"]) < .02 for row in rows)
        assert all(abs(row["errors"]["peak_interpolated_voltage_mv"]) > .5 for row in rows)
    assert "richardson_events" not in scored["scores"]["b1-fixed-027"]
    report = runner.decision_report(manifest, scored)
    assert report["gate"]["peak_sample_voltage_mv"] == .5 and "0.5 mV" in report["gate_amendment"]


def test_dt_series_manifest_is_a_halving_ladder_under_its_cap_with_identity_pairs():
    series = MANIFEST["dt_series"]
    arm_names = {arm["name"] for arm in MANIFEST["arms"]}
    new = [r for r in series["rungs"] if r["name"] not in arm_names]
    assert len(new) == 6 == series["cap"] and series["gate"] == {"rise_crossing_ms": 1.}
    assert {r["dt_ms"] for r in new} == {.0025, .00125, .000625} and {r["current_na"] for r in new} == {.19, .27}
    assert MANIFEST["identity_pairs"] == {"r-braincell-matched-027": "b1-fixed-027", "a1-matched-027": "b1-fixed-027"}
    assert MANIFEST["close"]["identity_basis"] == "r-braincell-matched-027"


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["dt_series"]["rungs"].append({"name": "extra", "current_na": .27, "dt_ms": .0003125, "reference": "027"}),
     "exceed the cap"),
    (lambda m: m["dt_series"]["rungs"][2].update(dt_ms=.003), "must halve dt"),
    (lambda m: m["dt_series"].pop("prediction"), "dt_series lacks prediction"),
    (lambda m: m["identity_pairs"].update({"a1-matched-019": "a1-matched-027"}), "different simulators"),
    (lambda m: m["identity_pairs"].update({"a1-matched-019": "nowhere"}), "different simulators"),
])
def test_dt_series_and_identity_violations_are_rejected(mutate, match):
    manifest = copy.deepcopy(MANIFEST)
    mutate(manifest)
    with pytest.raises(ValueError, match=match):
        runner.check_manifest(manifest)


def test_rise_summary_reads_the_last_event_and_the_one_millisecond_rule():
    gate = MANIFEST["dt_series"]["gate"]
    met = runner.rise_summary(runner.score_pair(_trace(), _trace(shift=-.8), (270., 329.5), gate))
    assert met["met"] and met["last_event_rise_error_ms"] == pytest.approx(-.8, abs=1e-6) and met["paired_events"] == 8
    over = runner.rise_summary(runner.score_pair(_trace(), _trace(shift=-1.2), (270., 329.5), gate))
    assert not over["met"] and over["max_abs_rise_error_ms"] == pytest.approx(1.2, abs=1e-6)
    fewer = runner.rise_summary(runner.score_pair(_trace(), _trace(drop=1), (270., 329.5), gate))
    assert not fewer["met"] and not fewer["count_equal"] and fewer["actual_count"] == 7


def test_adjacent_ratios_pair_only_consecutive_halvings_that_are_present():
    def rung(dt, last, maximum, events):
        rows = [{"errors": {"rise_crossing_ms": last*(k+1)/events}} for k in range(events)]
        return {"dt_ms": dt, "last_event_rise_error_ms": last, "max_abs_rise_error_ms": maximum,
                "paired_events": events, "events": rows}
    rungs = [rung(.005, 8., 8., 4), rung(.0025, 4., 4., 5), {"dt_ms": .00125, "status": "untested"},
             rung(.000625, 0., 1., 5)]
    ratios = runner.adjacent_ratios(rungs)
    assert len(ratios) == 1 and ratios[0]["dt_ms"] == .005 and ratios[0]["last_event_rise_error_ms_ratio"] == 2.
    assert ratios[0]["common_event"] == 4 and ratios[0]["common_event_rise_errors_ms"] == [8., 3.2]
    assert ratios[0]["common_event_rise_error_ratio"] == pytest.approx(2.5)
    rungs[2] = rung(.00125, 2., 2., 5)
    ratios = runner.adjacent_ratios(rungs)
    assert [r["half_dt_ms"] for r in ratios] == [.0025, .00125, .000625]
    assert ratios[-1]["last_event_rise_error_ms_ratio"] is None and ratios[-1]["max_abs_rise_error_ms_ratio"] == 2.
    assert ratios[-1]["common_event_rise_error_ratio"] is None
    assert runner.adjacent_ratios([rung(.005, 1., 1., 0), rung(.0025, 1., 1., 0)])[0]["common_event"] is None


def _series_tree(tmp_path, shifts):
    manifest = copy.deepcopy(MANIFEST)
    manifest["dt_series"]["window"] = "halving"
    for arm in manifest["arms"]:
        arm["window"] = "halving"
    out = tmp_path/"docs/evidence"/manifest["output_dir"]
    ref = tmp_path/"docs/evidence"/manifest["reference_dir"]
    out.mkdir(parents=True), ref.mkdir(parents=True)
    for stem in manifest["references"].values():
        np.savez(ref/(stem+".npz"), **_trace())
    for name, shift in shifts.items():
        np.savez(out/(name+".npz"), **_trace(shift))
    return manifest, out


def test_score_dt_series_names_the_coarsest_dt_met_at_both_inputs(tmp_path):
    manifest, _ = _series_tree(tmp_path, {"b1-fixed-027": 3.2, "b1-fixed-019": 3.2, "q-neuron-fixed-027-dt0025": 1.6,
                                          "q-neuron-fixed-019-dt0025": 1.6, "q-neuron-fixed-027-dt00125": .8,
                                          "q-neuron-fixed-019-dt00125": 1.1})
    series = runner.score_dt_series(tmp_path, manifest)
    assert series["dt_found_ms"] == "not reached within cap"
    assert series["met_at_both_inputs_by_dt"] == {"0.005": False, "0.0025": False, "0.00125": False, "0.000625": False}
    by_name = {r["name"]: r for r in series["rungs"]}
    assert by_name["q-neuron-fixed-027-dt000625"]["status"] == "untested"
    assert by_name["q-neuron-fixed-027-dt00125"]["met"] and not by_name["q-neuron-fixed-019-dt00125"]["met"]
    assert series["dt_found_by_input_ms"] == {"0.27": .00125, "0.19": "not reached within cap"}
    ratios = series["adjacent_ratios"]["0.27"]
    assert [r["dt_ms"] for r in ratios] == [.005, .0025]
    assert all(r["last_event_rise_error_ms_ratio"] == pytest.approx(2., abs=1e-4) for r in ratios)
    np.savez(tmp_path/"docs/evidence"/manifest["output_dir"]/"q-neuron-fixed-019-dt000625.npz", **_trace(.5))
    np.savez(tmp_path/"docs/evidence"/manifest["output_dir"]/"q-neuron-fixed-027-dt000625.npz", **_trace(.4))
    series = runner.score_dt_series(tmp_path, manifest)
    assert series["dt_found_ms"] == .000625 and set(series["inputs_sha256"]) == {r["name"] for r in series["rungs"]}
    assert runner.score_dt_series(tmp_path, {k: v for k, v in manifest.items() if k != "dt_series"}) is None


def test_score_identity_is_strict_and_marks_absent_partners_untested(tmp_path):
    manifest, out = _series_tree(tmp_path, {"b1-fixed-027": 0., "r-braincell-matched-027": 0., "a1-matched-027": 1e-3})
    identity = runner.score_identity(tmp_path, manifest)
    assert identity["r-braincell-matched-027"]["passed"] and identity["r-braincell-matched-027"]["reference"] == "b1-fixed-027"
    assert identity["r-braincell-matched-027"]["max_abs_peak_error_mv"] == 0.
    assert identity["r-braincell-matched-027"]["raw_voltage_max_abs_diff_mv"] == 0.
    assert identity["a1-matched-027"]["raw_voltage_max_abs_diff_mv"] > 0. and identity["a1-matched-027"]["max_abs_peak_sample_error_mv"] >= 0.
    assert not identity["a1-matched-027"]["passed"] and identity["a1-matched-027"]["first_failed_event"] == 7
    (out/"b1-fixed-027.npz").unlink()
    assert runner.score_identity(tmp_path, manifest)["a1-matched-027"]["status"] == "untested"


def test_close_report_reads_identity_from_the_basis_arm_and_carries_clocks(tmp_path):
    manifest, out = _series_tree(tmp_path, {"b1-fixed-027": 0., "b1-fixed-019": 0., "r-braincell-matched-027": 0.,
                                            "r-braincell-matched-027-halfdt": 0., "r-neuron-fixed-027-halfdt": 0.,
                                            "a1-matched-027": 0., "a1-matched-019": 0.})
    (out/"b1-fixed-027.timing.json").write_text(json.dumps({"wall_clock_s": 232.4}))
    report = runner.decision_report(manifest, runner.score_manifest(tmp_path, manifest))
    close = runner.close_report(manifest, report, runner.score_dt_series(tmp_path, manifest),
                                runner.score_identity(tmp_path, manifest), runner.wall_clocks(tmp_path, manifest))
    assert not close["identity_gate"]["closed"] and close["identity_gate"]["basis_arm"] == "r-braincell-matched-027"
    assert close["identity_gate"]["basis_measured"]["rise_crossing_ms"] == 0. and close["identity_gate"]["confirming_prediction_held"]
    assert set(close["identity_gate"]["confirming_full_train"]) == {"a1-matched-027"}
    assert close["dt_qualification"]["dt_found_ms"] == .005 and close["decision"] == "mesh"
    assert close["wall_clocks"] == {"b1-fixed-027": {"wall_clock_s": 232.4}}
    assert set(close["full_train_vs_cvode"]) == {"a1-matched-019", "a1-matched-027", "b1-fixed-019", "b1-fixed-027"}
    manifest["close"]["identity_values"] = {"rise_crossing_ms": 0., "peak_interpolated_voltage_mv": 0., "time_above_threshold_ms": 0.}
    close = runner.close_report(manifest, report, None, runner.score_identity(tmp_path, manifest), {})
    assert close["identity_gate"]["closed"]
    (out/"r-braincell-matched-027.npz").unlink()
    close = runner.close_report(manifest, report, None, runner.score_identity(tmp_path, manifest), {})
    assert not close["identity_gate"]["closed"] and close["dt_qualification"] is None


def test_main_close_writes_the_close_decision_beside_the_decision_json(tmp_path, monkeypatch, capsys):
    manifest, out = _series_tree(tmp_path, {"b1-fixed-027": 0., "b1-fixed-019": 0., "r-braincell-matched-027": 0.})
    (tmp_path/"docs/evidence"/manifest["mesh_source"]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path/"docs/evidence"/manifest["mesh_source"]).write_text((FOLDER/MANIFEST["mesh_source"]).read_text())
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "__file__", str(tmp_path/"docs/evidence/runner.py"))
    runner.main(["--manifest", str(path), "--score", "--close", "--peak-method", "interpolated"])
    close = json.loads((out/"sp2-close-decision.json").read_text())
    assert not close["identity_gate"]["closed"] and close["peak_method"] == "interpolated"
    assert close["identity_gate"]["basis"]["passed"] and close["dt_qualification"]["dt_found_ms"] == .005
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == {"identity_closed": False, "dt_found_ms": .005}


def test_reproduces_compares_at_two_significant_figures():
    quoted = {"rise_crossing_ms": 1.2e-9, "peak_interpolated_voltage_mv": 5.4e-6, "time_above_threshold_ms": 1.9e-11}
    assert runner.reproduces({"rise_crossing_ms": 1.2048e-9, "peak_interpolated_voltage_mv": 5.424e-6,
                              "time_above_threshold_ms": 1.9099e-11}, quoted)
    assert not runner.reproduces({"rise_crossing_ms": 1.3e-9, "peak_interpolated_voltage_mv": 5.424e-6,
                                  "time_above_threshold_ms": 1.9099e-11}, quoted)
    assert not runner.reproduces({"rise_crossing_ms": None}, quoted)
