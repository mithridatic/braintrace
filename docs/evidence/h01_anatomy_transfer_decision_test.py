"""Tests for the anatomy-transfer decision assembler."""

import json
import sys

import pytest

from docs.evidence import h01_anatomy_transfer_decision as decision


def _run(count, human, repeats=None, finite=True, donor_fit=None):
    verdict = decision.__dict__  # noqa: F841 - keep module reference explicit for readers
    difference = abs(count-human)
    exact = "held" if difference == 0 else ("missed_within_one" if difference <= 1 else "rejected")
    band = None if not repeats else ("held" if min(repeats) <= count <= max(repeats) else "missed")
    return dict(cell="c", current_na=.1, registered_human_count=human, repeat_counts=repeats,
                donor_model_count_on_own_anatomy=donor_fit, rest_before_pulse_mv=-80., peak_mv=20.,
                output_site=dict(count=count, finite=finite), soma_site=dict(count=count, finite=finite),
                verdict_vs_human=dict(exact=exact, repeat_band=band))


def test_donor_verdict_rules():
    assert decision.donor_verdict(_run(13, 12), _run(13, 12))["passed"] is True
    assert decision.donor_verdict(_run(14, 12), _run(14, 12))["passed"] is False
    assert decision.donor_verdict(_run(13, 14, (14, 14, 13, 12)), _run(13, 14, (14, 14, 13, 12)))["passed"] is True
    assert decision.donor_verdict(_run(11, 14, (14, 14, 13, 12)), _run(11, 14, (14, 14, 13, 12)))["passed"] is False
    assert decision.donor_verdict(_run(12, 12), None)["passed"] is False           # no dt-half confirmation
    assert decision.donor_verdict(_run(12, 12), _run(11, 12))["passed"] is False   # dt-half disagrees
    assert decision.donor_verdict(_run(12, 12, finite=False), _run(12, 12))["passed"] is False


def test_decide_fraction_and_hashes(tmp_path):
    for donor, label in decision.PRIMARY.items():
        folder = tmp_path/label
        folder.mkdir()
        good = label == "transfer-l4-090"
        (folder/"run.json").write_text(json.dumps(_run(12 if good else 2, 12)))
        half = tmp_path/(label+"-dthalf")
        half.mkdir()
        (half/"run.json").write_text(json.dumps(_run(12 if good else 2, 12)))
    result = decision.decide(tmp_path)
    assert result["anatomy_transfer"] == .25 and result["measured"] is True
    assert result["donors"]["l4-pyramidal-allen-527952884"]["passed"] is True
    assert sum(row["passed"] for row in result["donors"].values()) == 1
    assert len(result["input_hashes"]) == 8 and all(len(h) == 64 for h in result["input_hashes"].values())


REST = -81.


def _cell_run(count=12, human=12, cell="c", donor="l4-pyramidal-allen-527952884", finite=True, soma_finite=True,
              rest=REST, ret=REST, ret_available=True, pre=0):
    return dict(cell=cell, donor=donor, polarity="E", registered_human_count=human,
                output_site=dict(count=count, finite=finite), soma_site=dict(count=count, finite=soma_finite),
                pre_pulse_count=pre, rest_mean_mv=rest, return_mv=ret, return_available=ret_available,
                post_pulse_count=0)


def test_keep_verdict_all_rules_hold():
    verdict = decision.keep_verdict(_cell_run(), _cell_run(), REST)
    assert verdict["keep"] is True and verdict["drop_reason"] is None
    assert all(verdict["rules"].values()) and verdict["dt_half_count"] == 12
    assert verdict["count_band"] == [9, 15]


@pytest.mark.parametrize("primary, half, reason", [
    (_cell_run(finite=False), _cell_run(), "finite"),
    (_cell_run(soma_finite=False), _cell_run(), "finite"),
    (_cell_run(rest=REST-10.5), _cell_run(), "rest_and_return"),
    (_cell_run(ret=REST+10.5), _cell_run(), "rest_and_return"),
    (_cell_run(ret_available=False), _cell_run(), "rest_and_return"),
    (_cell_run(ret=None, ret_available=False), _cell_run(), "rest_and_return"),
    (_cell_run(pre=1), _cell_run(), "no_spike_before_pulse"),
    (_cell_run(count=16), _cell_run(count=16), "count_in_band"),
    (_cell_run(count=8), _cell_run(count=8), "count_in_band"),
    (_cell_run(), _cell_run(count=11), "dt_half_reproduces"),
    (_cell_run(), _cell_run(finite=False), "dt_half_reproduces"),
    (_cell_run(), None, "dt_half_missing"),
])
def test_keep_verdict_each_rule_failing_alone(primary, half, reason):
    verdict = decision.keep_verdict(primary, half, REST)
    assert verdict["keep"] is False and verdict["drop_reason"] == reason


def test_keep_verdict_rest_tolerance_is_inclusive():
    assert decision.keep_verdict(_cell_run(rest=REST+10., ret=REST-10.), _cell_run(), REST)["keep"] is True


def test_keep_verdict_reasons_join_in_rule_order():
    verdict = decision.keep_verdict(_cell_run(count=20, pre=2, finite=False), None, REST)
    assert verdict["drop_reason"] == "finite,no_spike_before_pulse,count_in_band"   # no repeat is owed to it
    assert verdict["rules"]["dt_half_reproduces"] is None


@pytest.mark.parametrize("human, band", [(10, [7, 13]), (12, [9, 15]), (14, [10, 18])])
def test_count_band(human, band):
    assert decision.count_band(human) == band
    lo, hi = band
    assert decision.keep_verdict(_cell_run(count=lo, human=human), _cell_run(count=lo), REST)["rules"]["count_in_band"]
    assert decision.keep_verdict(_cell_run(count=hi, human=human), _cell_run(count=hi), REST)["rules"]["count_in_band"]
    assert not decision.keep_verdict(_cell_run(count=lo-1, human=human), _cell_run(count=lo-1), REST)["rules"]["count_in_band"]
    assert not decision.keep_verdict(_cell_run(count=hi+1, human=human), _cell_run(count=hi+1), REST)["rules"]["count_in_band"]


def _population(tmp_path, with_half):
    l4, l2 = "l4-pyramidal-allen-527952884", "l2-pyramidal-allen-541563728"
    types = tmp_path/"types.json"
    types.write_text(json.dumps(dict(rows=[dict(cell_id="200", donor_key=l4, polarity="E"),
                                          dict(cell_id="100", donor_key=l2, polarity="E"),
                                          dict(cell_id="300", donor_key=l2, polarity="E")])))
    rests = tmp_path/"rests.json"
    rests.write_text(json.dumps(dict(donors={l4: dict(rest_mv=-81.), l2: dict(rest_mv=-84.)})))
    runs = tmp_path/"runs"
    (runs/"transfer-all-200").mkdir(parents=True)
    (runs/"transfer-all-200"/"run.json").write_text(json.dumps(_cell_run(cell="200", rest=-81., ret=-81.)))
    if with_half:
        (runs/"transfer-all-200-dthalf").mkdir()
        (runs/"transfer-all-200-dthalf"/"run.json").write_text(json.dumps(_cell_run(cell="200", rest=-81., ret=-81.)))
    (runs/"transfer-all-100").mkdir()
    (runs/"transfer-all-100"/"run.json").write_text(json.dumps(_cell_run(cell="100", donor=l2, count=2, human=10,
                                                                          rest=-84., ret=-27.)))
    return runs, types, rests


def test_decide_keep_primary_phase_lists_candidates_without_dt_half(tmp_path):
    runs, types, rests = _population(tmp_path, with_half=False)
    result = decision.decide_keep(runs, types, rests, phase="primary")
    assert result["candidates"] == ["200"]
    assert result["dropped"] == {"100": "rest_and_return,count_in_band"}
    assert result["missing"] == ["300"]
    assert "kept" not in result and set(result["input_hashes"]) == {"transfer-all-200", "transfer-all-100"}


def test_decide_keep_final_phase_needs_dt_half(tmp_path):
    runs, types, rests = _population(tmp_path, with_half=False)
    result = decision.decide_keep(runs, types, rests, phase="final")
    assert result["kept"] == [] and result["dropped"]["200"] == "dt_half_missing"
    assert result["counts"] == dict(kept=0, dropped=2, missing=1, total=3)


def test_decide_keep_final_phase(tmp_path):
    runs, types, rests = _population(tmp_path, with_half=True)
    result = decision.decide_keep(runs, types, rests, phase="final")
    assert result["kept"] == ["200"]
    assert result["dropped"] == {"100": "rest_and_return,count_in_band"}
    assert result["missing"] == ["300"]
    assert result["counts"] == dict(kept=1, dropped=1, missing=1, total=3)
    assert result["by_donor"] == {"l4-pyramidal-allen-527952884": dict(kept=1, dropped=0),
                                  "l2-pyramidal-allen-541563728": dict(kept=0, dropped=1)}
    assert set(result["input_hashes"]) == {"transfer-all-200", "transfer-all-200-dthalf", "transfer-all-100"}
    assert all(len(h) == 64 for h in result["input_hashes"].values())
    assert result["rule"] and result["scope"] and result["cells"]["200"]["keep"] is True


def test_keep_cli_writes_decision(tmp_path, capsys):
    runs, types, rests = _population(tmp_path, with_half=True)
    output = tmp_path/"decision.json"
    argv = sys.argv
    sys.argv = ["x", "--folder", str(runs), "--output", str(output), "--keep", "--types", str(types),
                "--rests", str(rests), "--phase", "final"]
    try:
        decision.main()
    finally:
        sys.argv = argv
    written = json.loads(output.read_text())
    assert written["kept"] == ["200"]
    out = capsys.readouterr().out
    assert "kept 1 / dropped 1 / missing 1 of 3" in out and "drop 100: rest_and_return,count_in_band" in out


DONOR = dict(rest_mv=REST, sd_mv=.02, rest_repeat_sd_mv=.1, rheobase_pa=50., sweep_step_pa=20., highest_firing_pa=170.,
             repeat_counts=[12])


def _ramp(rheobase_na=.055, block_na=None, spikes=(1100., 1500., 1900., 2010.), ramp_max_na=.27, finite=True, pre=0):
    return dict(ramp_max_na=ramp_max_na, rheobase_na=rheobase_na, block_na=block_na,
                ramp_spike_times_ms=list(spikes), ramp_count=len(spikes), finite=finite, pre_pulse_count=pre)


def _cell_run_failure(**overrides):
    run = _cell_run(**{k: v for k, v in overrides.items() if k != "pulse"})
    run["pulse_ms"] = overrides.get("pulse", [1020., 2020.])
    return run


def test_failure_verdict_all_rules_hold_and_carries_the_legacy_verdict():
    verdict = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(), DONOR)
    assert verdict["keep"] is True and verdict["drop_reason"] is None
    assert all(v is True for v in verdict["rules"].values()) and verdict["legacy_verdict"]["keep"] is True
    assert verdict["rest_tolerance_mv"] == pytest.approx(.3)   # 3 x the across-sweep repeat sd, not the within-trace sd
    readings = verdict["readings"]
    assert readings["rest_sd_mv"] == .1 and readings["rest_sd_source"].startswith("across-sweep")
    assert readings["model_rheobase_pa"] == pytest.approx(55.) and readings["ramp_max_pa"] == pytest.approx(270.)
    assert readings["model_last_spike_pa"] == pytest.approx(267.3) and readings["block_above_recorded_range"] is None


@pytest.mark.parametrize("primary, ramp, half, reason", [
    (_cell_run_failure(finite=False), _ramp(), _cell_run_failure(), "finite"),
    (_cell_run_failure(), _ramp(finite=False), _cell_run_failure(), "finite"),
    (_cell_run_failure(), _ramp(rheobase_na=.0701), _cell_run_failure(), "rheobase_in_step"),
    (_cell_run_failure(), _ramp(rheobase_na=None, spikes=()), _cell_run_failure(), "rheobase_in_step,fires_at_highest"),
    (_cell_run_failure(), _ramp(block_na=.15, spikes=(1100., 1500.)), _cell_run_failure(), "fires_at_highest"),
    (_cell_run_failure(count=13), _ramp(), _cell_run_failure(count=13), "count_in_repeat_range"),
    (_cell_run_failure(rest=REST-.31), _ramp(), _cell_run_failure(), "rest_in_donor_spread"),
    (_cell_run_failure(ret=REST+.31), _ramp(), _cell_run_failure(), "rest_in_donor_spread"),
    (_cell_run_failure(ret_available=False), _ramp(), _cell_run_failure(), "rest_in_donor_spread"),
    (_cell_run_failure(pre=1), _ramp(), _cell_run_failure(), "no_spike_before_pulse"),
    (_cell_run_failure(), _ramp(pre=1), _cell_run_failure(), "no_spike_before_pulse"),
    (_cell_run_failure(), _ramp(), _cell_run_failure(count=11), "dt_half_reproduces"),
    (_cell_run_failure(), _ramp(), None, "dt_half_missing"),
    (_cell_run_failure(), None, _cell_run_failure(), "ramp_missing"),
])
def test_failure_verdict_each_rule_failing_alone(primary, ramp, half, reason):
    verdict = decision.keep_verdict_failure(primary, ramp, half, DONOR)
    assert verdict["keep"] is False and verdict["drop_reason"] == reason


def test_failure_verdict_block_inside_the_recorded_range_fails_and_above_it_is_recorded_only():
    below = decision.keep_verdict_failure(_cell_run_failure(), _ramp(block_na=.16, spikes=(1100., 1500., 1612.)),
                                          _cell_run_failure(), DONOR)
    assert below["rules"]["fires_at_highest"] is False and below["readings"]["block_above_recorded_range"] is False
    above = decision.keep_verdict_failure(_cell_run_failure(), _ramp(block_na=.2, spikes=(1100., 1500., 1800.)),
                                          _cell_run_failure(), DONOR)
    assert above["keep"] is True and above["readings"]["block_above_recorded_range"] is True
    assert above["readings"]["model_block_pa"] == pytest.approx(200.)


def test_failure_verdict_repeat_range_is_inclusive_and_single_repeat_is_exact():
    donor = dict(DONOR, repeat_counts=[14, 14, 13, 12])
    for count in (12, 13, 14):
        assert decision.keep_verdict_failure(_cell_run_failure(count=count), _ramp(), _cell_run_failure(count=count),
                                             donor)["rules"]["count_in_repeat_range"]
    assert not decision.keep_verdict_failure(_cell_run_failure(count=15), _ramp(), _cell_run_failure(count=15),
                                             donor)["rules"]["count_in_repeat_range"]
    assert not decision.keep_verdict_failure(_cell_run_failure(count=11), _ramp(), _cell_run_failure(count=11),
                                             DONOR)["rules"]["count_in_repeat_range"]


def test_failure_verdict_rest_falls_back_to_the_within_trace_sd_when_no_repeat_sd():
    donor = {k: v for k, v in DONOR.items() if k != "rest_repeat_sd_mv"}
    verdict = decision.keep_verdict_failure(_cell_run_failure(rest=REST+.05), _ramp(), _cell_run_failure(), donor)
    assert verdict["rest_tolerance_mv"] == pytest.approx(.06) and verdict["readings"]["rest_sd_mv"] == .02
    assert verdict["readings"]["rest_sd_source"].startswith("within-trace") and verdict["keep"] is True
    assert decision.keep_verdict_failure(_cell_run_failure(rest=REST+.07), _ramp(), _cell_run_failure(), donor)["keep"] is False


def test_failure_verdict_unavailable_datum_is_never_a_keep():
    short = decision.keep_verdict_failure(_cell_run_failure(), _ramp(ramp_max_na=.15), _cell_run_failure(), DONOR)
    assert short["rules"]["fires_at_highest"] is None and short["drop_reason"] == "firing_datum_unavailable"
    no_human = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(),
                                             dict(DONOR, rheobase_pa=None))
    assert no_human["rules"]["rheobase_in_step"] is None and no_human["keep"] is False
    no_repeats = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(),
                                               dict(DONOR, repeat_counts=[]))
    assert no_repeats["rules"]["count_in_repeat_range"] is None and no_repeats["keep"] is False


def _failure_population(tmp_path, with_ramp=True, with_half=True):
    l4 = "l4-pyramidal-allen-527952884"
    types = tmp_path/"types.json"
    types.write_text(json.dumps(dict(rows=[dict(cell_id="200", donor_key=l4, polarity="E"),
                                          dict(cell_id="300", donor_key=l4, polarity="E")])))
    rests = tmp_path/"rests.json"
    rests.write_text(json.dumps(dict(donors={l4: dict(rest_mv=REST, sd_mv=.02)})))
    datums = tmp_path/"datums.json"
    datums.write_text(json.dumps(dict(donors={l4: dict(rheobase_pa=50., sweep_step_pa=20., highest_firing_pa=170.,
                                                        repeat_counts=[12], rest_repeat_sd_mv=.1)})))
    runs = tmp_path/"runs"
    for cell, count in (("200", 12), ("300", 15)):
        (runs/f"transfer-all-{cell}").mkdir(parents=True)
        (runs/f"transfer-all-{cell}"/"run.json").write_text(json.dumps(_cell_run_failure(cell=cell, count=count)))
        if with_ramp:
            (runs/f"transfer-all-{cell}-ramp").mkdir()
            (runs/f"transfer-all-{cell}-ramp"/"run.json").write_text(json.dumps(dict(cell=cell, ramp=None, **_ramp())))
        if with_half:
            (runs/f"transfer-all-{cell}-dthalf").mkdir()
            (runs/f"transfer-all-{cell}-dthalf"/"run.json").write_text(json.dumps(_cell_run_failure(cell=cell, count=count)))
    return runs, types, rests, datums


def test_decide_keep_failure_gate_reads_ramps_and_datums(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path)
    result = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums)
    assert result["gate"] == "failure" and result["kept"] == ["200"]
    assert result["dropped"] == {"300": "count_in_repeat_range"} and result["rule"] == decision.FAILURE_RULE
    assert set(result["input_hashes"]) == {"transfer-all-200", "transfer-all-200-ramp", "transfer-all-200-dthalf",
                                           "transfer-all-300", "transfer-all-300-ramp", "transfer-all-300-dthalf"}
    assert result["cells"]["200"]["legacy_verdict"]["keep"] is True and len(result["datums_sha256"]) == 64
    assert result["cells"]["200"]["rest_tolerance_mv"] == pytest.approx(.3)   # datums repeat sd, not donor-rest sd
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == ["200"] and primary["dropped"] == {"300": "count_in_repeat_range"}


def test_decide_keep_failure_gate_without_ramp_or_datums(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path, with_ramp=False)
    result = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums)
    assert result["kept"] == [] and result["dropped"]["200"] == "ramp_missing"
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == [] and "rheobase_in_step" in primary["dropped"]["200"]
    with pytest.raises(ValueError, match="datums"):
        decision.decide_keep(runs, types, rests, gate="failure")


def test_bands_gate_is_unchanged_by_the_failure_gate(tmp_path):
    runs, types, rests, _ = _failure_population(tmp_path)
    result = decision.decide_keep(runs, types, rests)
    assert result["gate"] == "bands" and result["kept"] == ["200", "300"]   # 15 is inside the old band of 12
    assert "legacy_verdict" not in result["cells"]["200"]


def test_cli_failure_gate_writes_the_decision(tmp_path, monkeypatch, capsys):
    runs, types, rests, datums = _failure_population(tmp_path)
    output = tmp_path/"decision.json"
    monkeypatch.setattr(sys, "argv", ["decide", "--folder", str(runs), "--output", str(output), "--keep",
                                      "--types", str(types), "--rests", str(rests), "--gate", "failure",
                                      "--datums", str(datums)])
    decision.main()
    written = json.loads(output.read_text())
    assert written["gate"] == "failure" and written["kept"] == ["200"]
    assert "kept 1 / dropped 1 / missing 0 of 2" in capsys.readouterr().out
