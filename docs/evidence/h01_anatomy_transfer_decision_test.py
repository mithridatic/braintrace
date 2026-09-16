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
