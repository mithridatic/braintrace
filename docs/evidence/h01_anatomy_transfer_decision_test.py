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


AFTER = REST-.8    # the family's level over the 10 ms ending 200 ms after offset sits below its pre-pulse rest
TYPE = dict(population="spiny_l4", label="human spiny (pyramidal) layer 4", cells_with_matched_sweeps=37,
            tolerance_rule="2 sd across cells", count=dict(n=37, mean=12.75, sd=10.04, tolerance=20.08),
            rest_mv=dict(n=37, mean=-82.18, sd=4.15, tolerance=8.30), after_mv=dict(n=37, mean=-82., sd=4.10, tolerance=8.20),
            windows=dict(count="num_spikes", rest_mv="pre_vm_mv", after_mv="post_vm_mv"))
DONOR = dict(rest_mv=REST-.4, sd_mv=.02, rest_repeat_mean_mv=REST, rest_repeat_sd_mv=.1, rheobase_pa=50., sweep_step_pa=20.,
             highest_firing_pa=170., repeat_counts=[12], measured_counts_at_primary=[12],
             after_repeat_mean_mv=AFTER, after_repeat_sd_mv=.2,
             count_band=dict(min=11, max=15, counts=[11, 12, 15], sweeps=[38, 39, 40], amplitudes_pa=[70., 90., 110.]),
             type_population=TYPE,
             allen_ramp_threshold=dict(specimen_id=527952884, threshold_i_long_square_pa=50., threshold_i_ramp_pa=52.08,
                                       peak_t_ramp_s=3.11, step_to_ramp_offset_pa=2.08))


def _ramp(rheobase_na=.055, block_na=None, spikes=(1100., 1500., 1900., 2010.), ramp_max_na=.27, finite=True, pre=0):
    """A ramp run.json in the runner's layout: the readings under ``ramp``, the windows at the top level."""
    readings = dict(ramp_max_na=ramp_max_na, rheobase_na=rheobase_na, block_na=block_na,
                    ramp_spike_times_ms=list(spikes), ramp_count=len(spikes), finite=finite)
    return dict(drive="ramp", ramp_max_na=ramp_max_na, current_na=None, pulse_ms=[1020., 2020.], pre_pulse_count=pre,
                output_site=dict(count=len(spikes), finite=finite), soma_site=dict(count=len(spikes), finite=finite),
                ramp=dict(readings, soma_site=dict(readings)), rest_mean_mv=REST, return_mv=AFTER, return_available=True)


def test_failure_verdict_reads_the_runner_layout_of_a_ramp_run():
    """The runner (h01_anatomy_transfer_run.summarize) nests the ramp readings under ``ramp``; a ramp that fires to
    its end must not read as silent (the defect seen on the first six C3 candidates, 2026-09-16)."""
    run = pytest.importorskip("h01_anatomy_transfer_run")
    import numpy as np
    from types import SimpleNamespace
    time = np.arange(.01, 2300.01, .01)
    voltage = np.full_like(time, REST)
    for centre in np.arange(1100., 2020., 40.):    # 23 spikes from 80 pA to the ramp end
        voltage[(time >= centre) & (time < centre+.5)] = 20.
    args = SimpleNamespace(cell="c", donor="l4-pyramidal-allen-527952884", polarity="E", current_na=None, ramp_na=.27,
                           pulse_on_ms=1020., pulse_ms=1000., duration_ms=2300., dt_ms=.01, solver="s", max_cv_um=10.,
                           registered_count=None, repeat_counts=None, donor_model_count=None, tags=None, cell_table=None)
    voltage[(time >= 2210.) & (time <= 2220.)] = AFTER
    summary = run.summarize(args, dict(time_ms=time, output_voltage=voltage, voltage=voltage),
                            dict(component=0, source_sha256="x"*64, component_nodes=1, n_compartments=1))
    assert summary["ramp"]["rheobase_na"] is not None and "rheobase_na" not in summary   # the layout under test
    verdict = decision.keep_verdict_failure(_cell_run_failure(), summary, _cell_run_failure(), DONOR)
    assert verdict["recorded_readings"]["rheobase_in_step"] is False    # 21.6 pA rheobase vs human 50 +- 20, recorded only
    assert verdict["hard_rules"]["recruitable_in_human_range"] is True
    assert verdict["hard_rules"]["fires_at_highest"] is True and verdict["hard_rules"]["no_block_in_recorded_range"] is True
    assert verdict["readings"]["model_rheobase_pa"] == pytest.approx(21.6, abs=.01) and verdict["readings"]["model_last_spike_pa"] > 170.
    assert verdict["rules"]["finite"] is True and verdict["rules"]["no_spike_before_pulse"] is True


def _cell_run_failure(**overrides):
    overrides.setdefault("ret", AFTER)
    run = _cell_run(**{k: v for k, v in overrides.items() if k != "pulse"})
    run["pulse_ms"] = overrides.get("pulse", [1020., 2020.])
    return run


def test_failure_verdict_all_rules_hold_and_carries_the_comparison_columns():
    verdict = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(), DONOR)
    assert verdict["keep"] is True and verdict["drop_reason"] is None and verdict["pending"] is None
    assert tuple(verdict["rules"]) == decision.FAILURE_RULE_NAMES and all(v is True for v in verdict["rules"].values())
    assert tuple(verdict["hard_rules"]) == decision.HARD_RULE_NAMES
    assert tuple(verdict["plausibility_rules"]) == decision.PLAUSIBILITY_RULE_NAMES
    assert verdict["donor_band_rules"] == dict(count_in_repeat_range=True, rest_in_donor_spread=True)
    assert verdict["donor_band_failed"] == [] and verdict["legacy_verdict"]["keep"] is True
    readings = verdict["readings"]
    assert readings["count_datum"] == 12. and readings["count_tolerance"] == 20.08 and readings["type_population"] == "spiny_l4"
    assert readings["rest_type_datum_mv"] == REST and readings["rest_type_tolerance_mv"] == 8.3
    assert readings["after_type_datum_mv"] == AFTER and readings["after_type_tolerance_mv"] == 8.2
    assert readings["type_cells"] == 37 and readings["type_tolerance_rule"] == "2 sd across cells"
    assert verdict["rest_tolerance_mv"] == pytest.approx(.3)   # single-donor family band, comparison column
    assert readings["human_count_band"] == [11, 15] and readings["model_rheobase_pa"] == pytest.approx(55.)
    assert readings["block_above_recorded_range"] is None and readings["return_leg_scored"] is True
    assert verdict["recorded_readings"] == dict(rheobase_in_step=True) and readings["rheobase_in_step"] is True
    assert readings["human_threshold_i_ramp_pa"] == 52.08 and readings["human_step_to_ramp_offset_pa"] == 2.08
    assert readings["model_ramp_slope_pa_per_s"] == pytest.approx(270.) and readings["fires_at_primary_step"] is True


def test_rheobase_in_step_is_recorded_not_scored_and_recruitable_uses_the_step_run():
    # ramp rheobase 169.9 pA: outside the human step window 50 +- 20 but at or below the highest amplitude (170) -> keep
    wide = decision.keep_verdict_failure(_cell_run_failure(), _ramp(rheobase_na=.1699, spikes=(1650., 1900., 2010.)),
                                         _cell_run_failure(), DONOR)
    assert wide["keep"] is True and wide["recorded_readings"]["rheobase_in_step"] is False
    assert wide["hard_rules"]["recruitable_in_human_range"] is True and "rheobase_in_step" not in wide["rules"]
    # a cell whose ramp fires but whose step run at the primary drive is silent is not recruitable
    silent_step = decision.keep_verdict_failure(_cell_run_failure(count=0), _ramp(), _cell_run_failure(count=0), DONOR)
    assert silent_step["hard_rules"]["recruitable_in_human_range"] is False and silent_step["readings"]["fires_at_primary_step"] is False
    no_allen = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(),
                                             {k: v for k, v in DONOR.items() if k != "allen_ramp_threshold"})
    assert no_allen["keep"] is True and no_allen["readings"]["human_threshold_i_ramp_pa"] is None
    pending = decision.keep_verdict_failure(_cell_run_failure(), None, _cell_run_failure(), DONOR)
    assert pending["hard_rules"]["recruitable_in_human_range"] is None and pending["pending"] == "ramp_missing"


@pytest.mark.parametrize("primary, ramp, half, reason", [
    (_cell_run_failure(finite=False), _ramp(), _cell_run_failure(), "finite"),
    (_cell_run_failure(), _ramp(finite=False), _cell_run_failure(), "finite"),
    (_cell_run_failure(pre=1), _ramp(), _cell_run_failure(), "no_spike_before_pulse"),
    (_cell_run_failure(), _ramp(pre=1), _cell_run_failure(), "no_spike_before_pulse"),
    (_cell_run_failure(), _ramp(rheobase_na=.1701, spikes=(1650., 1900., 2010.)), _cell_run_failure(), "recruitable_in_human_range"),
    (_cell_run_failure(count=0), _ramp(), _cell_run_failure(count=0), "recruitable_in_human_range"),
    (_cell_run_failure(), _ramp(rheobase_na=None, spikes=()), _cell_run_failure(), "recruitable_in_human_range,fires_at_highest"),
    (_cell_run_failure(), _ramp(block_na=.15, spikes=(1100., 1500.)), _cell_run_failure(),
     "fires_at_highest,no_block_in_recorded_range"),
    (_cell_run_failure(), _ramp(), _cell_run_failure(count=11), "dt_half_reproduces"),
    (_cell_run_failure(count=33), _ramp(), _cell_run_failure(count=33), "count_in_type_spread"),
    (_cell_run_failure(rest=REST-8.31), _ramp(), _cell_run_failure(), "rest_in_type_spread"),
    (_cell_run_failure(ret=AFTER+8.21), _ramp(), _cell_run_failure(), "return_in_type_spread"),
    (_cell_run_failure(ret_available=False), _ramp(), _cell_run_failure(), "return_in_type_spread"),
])
def test_failure_verdict_each_rule_failing_alone(primary, ramp, half, reason):
    verdict = decision.keep_verdict_failure(primary, ramp, half, DONOR)
    assert verdict["keep"] is False and verdict["drop_reason"] == reason and verdict["pending"] is None


def test_failure_verdict_plausibility_is_the_type_spread_and_the_donor_band_is_only_recorded():
    # count 32: inside 12 +- 20.08, outside the single-donor band [11, 15] -> keep, donor band failed, recorded
    wide = decision.keep_verdict_failure(_cell_run_failure(count=32), _ramp(), _cell_run_failure(count=32), DONOR)
    assert wide["keep"] is True and wide["donor_band_rules"]["count_in_repeat_range"] is False
    assert wide["donor_band_failed"] == ["count_in_repeat_range"] and wide["legacy_verdict"]["rules"]["count_in_band"] is False
    # return 5 mV below the after datum: inside the type spread (8.2), outside the donor family band (0.6)
    low = decision.keep_verdict_failure(_cell_run_failure(ret=AFTER-5.), _ramp(), _cell_run_failure(), DONOR)
    assert low["keep"] is True and low["donor_band_rules"]["rest_in_donor_spread"] is False
    assert low["readings"]["return_leg_held"] is False and low["rules"]["return_in_type_spread"] is True
    # the datum is the donor's own value, not the type mean: count 12 with the type mean at 12.75
    assert wide["readings"]["count_datum"] == 12. and wide["readings"]["type_count_mean"] == 12.75
    # SST-style repeats: the datum is the mean of the donor's measured counts at the primary
    sst = dict(DONOR, measured_counts_at_primary=[14, 14, 13, 12], repeat_counts=[14, 14, 13, 12])
    assert decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(), sst)["readings"]["count_datum"] == 13.25


def test_failure_verdict_block_edge_is_a_hard_failure_inside_the_recorded_range_and_recorded_above_it():
    below = decision.keep_verdict_failure(_cell_run_failure(), _ramp(block_na=.16, spikes=(1100., 1500., 1612.)),
                                          _cell_run_failure(), DONOR)
    assert below["hard_rules"]["no_block_in_recorded_range"] is False and below["hard_rules"]["fires_at_highest"] is False
    assert below["readings"]["block_above_recorded_range"] is False and below["keep"] is False
    at_edge = decision.keep_verdict_failure(_cell_run_failure(), _ramp(block_na=.17, spikes=(1100., 1500., 1650.)),
                                            _cell_run_failure(), DONOR)
    assert at_edge["hard_rules"]["no_block_in_recorded_range"] is False   # at the highest amplitude is inside the range
    above = decision.keep_verdict_failure(_cell_run_failure(), _ramp(block_na=.2, spikes=(1100., 1500., 1800.)),
                                          _cell_run_failure(), DONOR)
    assert above["keep"] is True and above["readings"]["block_above_recorded_range"] is True
    assert above["hard_rules"]["no_block_in_recorded_range"] is True and above["readings"]["model_block_pa"] == pytest.approx(200.)


def test_failure_verdict_missing_runs_are_pending_not_dropped():
    no_half = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), None, DONOR)
    assert no_half["keep"] is False and no_half["drop_reason"] is None and no_half["pending"] == "dt_half_missing"
    assert no_half["unmeasured_rules"] == ["dt_half_reproduces"]
    no_ramp = decision.keep_verdict_failure(_cell_run_failure(), None, _cell_run_failure(), DONOR)
    assert no_ramp["keep"] is False and no_ramp["drop_reason"] is None and no_ramp["pending"] == "ramp_missing"
    assert no_ramp["unmeasured_rules"] == ["recruitable_in_human_range", "fires_at_highest", "no_block_in_recorded_range"]
    assert no_ramp["rules"]["count_in_type_spread"] is True   # read from the primary run, ramp or not
    # a measured failure is a drop even while the ramp is pending
    both = decision.keep_verdict_failure(_cell_run_failure(count=40), None, _cell_run_failure(count=40), DONOR)
    assert both["drop_reason"] == "count_in_type_spread" and both["pending"] == "ramp_missing" and both["keep"] is False


def test_failure_verdict_unavailable_datum_is_never_a_keep():
    short = decision.keep_verdict_failure(_cell_run_failure(), _ramp(ramp_max_na=.15), _cell_run_failure(), DONOR)
    assert short["rules"]["fires_at_highest"] is None and short["drop_reason"] == "firing_datum_unavailable"
    no_human = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(), dict(DONOR, highest_firing_pa=None))
    assert no_human["rules"]["recruitable_in_human_range"] is None and no_human["keep"] is False
    no_step = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(), dict(DONOR, rheobase_pa=None))
    assert no_step["readings"]["rheobase_in_step"] is None and no_step["keep"] is True   # a recorded reading, not a rule
    no_type = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(),
                                            {k: v for k, v in DONOR.items() if k != "type_population"})
    assert all(no_type["rules"][n] is None for n in decision.PLAUSIBILITY_RULE_NAMES)
    assert no_type["drop_reason"] == "plausibility_datum_unavailable" and no_type["keep"] is False
    assert no_type["readings"]["type_population"] is None and no_type["donor_band_rules"]["count_in_repeat_range"] is True
    no_count = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(),
                                             dict(DONOR, measured_counts_at_primary=[], repeat_counts=[]))
    assert no_count["rules"]["count_in_type_spread"] is None and no_count["readings"]["count_datum"] is None
    no_after = decision.keep_verdict_failure(_cell_run_failure(), _ramp(), _cell_run_failure(),
                                             dict(DONOR, after_repeat_mean_mv=None, after_repeat_sd_mv=None))
    assert no_after["rules"]["return_in_type_spread"] is None and no_after["keep"] is False
    assert no_after["donor_band_rules"]["rest_in_donor_spread"] is True and no_after["readings"]["return_leg_scored"] is False


def test_donor_band_columns_keep_their_own_datums():
    # single-donor count band [11, 15] is inclusive; the family rest bands are 3 across-sweep sd around each leg's own datum
    for count in (11, 15):
        assert decision.keep_verdict_failure(_cell_run_failure(count=count), _ramp(), _cell_run_failure(count=count),
                                             DONOR)["donor_band_rules"]["count_in_repeat_range"] is True
    at_rest = decision.keep_verdict_failure(_cell_run_failure(ret=REST), _ramp(), _cell_run_failure(), DONOR)
    assert at_rest["donor_band_rules"]["rest_in_donor_spread"] is False and at_rest["readings"]["rest_leg_held"] is True
    assert at_rest["legacy_verdict"]["rules"]["rest_and_return"] is True and at_rest["keep"] is True
    single = {k: v for k, v in DONOR.items() if k not in ("rest_repeat_sd_mv", "rest_repeat_mean_mv")}
    fallback = decision.keep_verdict_failure(_cell_run_failure(rest=REST-.4+.05), _ramp(), _cell_run_failure(), single)
    assert fallback["rest_tolerance_mv"] == pytest.approx(.06) and fallback["readings"]["rest_sd_source"].startswith("within-trace")
    assert fallback["readings"]["rest_datum_mv"] == REST-.4 and fallback["readings"]["rest_type_datum_mv"] is None
    assert fallback["rules"]["rest_in_type_spread"] is None   # the type rule needs the family datum


def _failure_population(tmp_path, with_ramp=True, with_half=True):
    l4 = "l4-pyramidal-allen-527952884"
    types = tmp_path/"types.json"
    types.write_text(json.dumps(dict(rows=[dict(cell_id="200", donor_key=l4, polarity="E"),
                                          dict(cell_id="300", donor_key=l4, polarity="E")])))
    rests = tmp_path/"rests.json"
    rests.write_text(json.dumps(dict(donors={l4: dict(rest_mv=REST-.4, sd_mv=.02)})))
    datums = tmp_path/"datums.json"
    datums.write_text(json.dumps(dict(donors={l4: dict(rheobase_pa=50., sweep_step_pa=20., highest_firing_pa=170.,
                                                        repeat_counts=[12], measured_counts_at_primary=[12],
                                                        rest_repeat_mean_mv=REST, rest_repeat_sd_mv=.1,
                                                        after_repeat_mean_mv=AFTER, after_repeat_sd_mv=.2,
                                                        count_band=dict(min=12, max=14, counts=[12, 14], sweeps=[39, 40]),
                                                        type_population=TYPE)})))
    runs = tmp_path/"runs"
    for cell, count in (("200", 12), ("300", 40)):
        (runs/f"transfer-all-{cell}").mkdir(parents=True)
        (runs/f"transfer-all-{cell}"/"run.json").write_text(json.dumps(_cell_run_failure(cell=cell, count=count)))
        if with_ramp:
            (runs/f"transfer-all-{cell}-ramp").mkdir()
            (runs/f"transfer-all-{cell}-ramp"/"run.json").write_text(json.dumps(dict(cell=cell, **_ramp())))
        if with_half:
            (runs/f"transfer-all-{cell}-dthalf").mkdir()
            (runs/f"transfer-all-{cell}-dthalf"/"run.json").write_text(json.dumps(_cell_run_failure(cell=cell, count=count)))
    return runs, types, rests, datums


def test_decide_keep_failure_gate_reads_ramps_and_datums(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path)
    result = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums)
    assert result["gate"] == "failure" and result["kept"] == ["200"]
    assert result["dropped"] == {"300": "count_in_type_spread"} and result["rule"] == decision.FAILURE_RULE
    assert set(result["input_hashes"]) == {"transfer-all-200", "transfer-all-200-ramp", "transfer-all-200-dthalf",
                                           "transfer-all-300", "transfer-all-300-ramp", "transfer-all-300-dthalf"}
    assert result["cells"]["200"]["legacy_verdict"]["keep"] is True and len(result["datums_sha256"]) == 64
    counts = result["rule_counts"]
    assert counts["hard"]["finite"] == dict(held=2, failed=0, unmeasured=0)
    assert counts["plausibility"]["count_in_type_spread"] == dict(held=1, failed=1, unmeasured=0)
    assert counts["donor_band"]["count_in_repeat_range"] == dict(held=1, failed=1, unmeasured=0)
    assert counts["legacy"]["count_in_band"] == dict(held=1, failed=1, unmeasured=0)
    assert counts["recorded"]["rheobase_in_step"] == dict(held=2, failed=0, unmeasured=0)
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == ["200"] and primary["dropped"] == {"300": "count_in_type_spread"}


def test_decide_keep_failure_gate_without_ramp_or_datums(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path, with_ramp=False)
    result = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums)
    assert result["kept"] == [] and result["pending"] == {"200": "ramp_missing"}
    assert result["dropped"] == {"300": "count_in_type_spread"}   # a measured failure drops before the ramp runs
    assert result["counts"] == dict(kept=0, dropped=1, missing=0, total=2, pending=1)
    assert result["cells"]["200"]["rules"]["recruitable_in_human_range"] is None and result["cells"]["200"]["keep"] is False
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == [] and primary["pending"] == {"200": "ramp_missing"}
    with pytest.raises(ValueError, match="datums"):
        decision.decide_keep(runs, types, rests, gate="failure")


def test_decide_keep_primary_phase_drops_a_cell_whose_ramp_fails_so_no_repeat_is_spent(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path, with_ramp=False, with_half=False)
    blocked = _ramp(block_na=.15, spikes=(1100., 1500.))   # stops firing below the human's highest amplitude
    (runs/"transfer-all-200-ramp").mkdir()
    (runs/"transfer-all-200-ramp"/"run.json").write_text(json.dumps(dict(cell="200", **blocked)))
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == [] and primary["pending"] == {}
    assert primary["dropped"] == {"200": "fires_at_highest,no_block_in_recorded_range", "300": "count_in_type_spread"}
    silent = _ramp(rheobase_na=None, spikes=())
    (runs/"transfer-all-200-ramp"/"run.json").write_text(json.dumps(dict(cell="200", **silent)))
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["dropped"]["200"] == "recruitable_in_human_range,fires_at_highest" and primary["candidates"] == []


def test_decide_keep_primary_phase_never_lists_a_cell_with_an_unavailable_datum(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path, with_half=False)
    doc = json.loads(datums.read_text())
    del doc["donors"]["l4-pyramidal-allen-527952884"]["type_population"]
    datums.write_text(json.dumps(doc))
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == [] and primary["dropped"]["200"] == "plausibility_datum_unavailable"
    doc["donors"]["l4-pyramidal-allen-527952884"]["type_population"] = TYPE
    doc["donors"]["l4-pyramidal-allen-527952884"]["highest_firing_pa"] = None
    datums.write_text(json.dumps(doc))
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == [] and primary["dropped"]["200"] == "firing_datum_unavailable"


def test_decide_keep_failure_gate_dt_half_missing_is_pending_in_the_final_phase_only(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path, with_half=False)
    final = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums)
    assert final["kept"] == [] and final["pending"] == {"200": "dt_half_missing"}
    primary = decision.decide_keep(runs, types, rests, phase="primary", gate="failure", datums_path=datums)
    assert primary["candidates"] == ["200"] and primary["pending"] == {}   # the follower launches its repeat


def test_decide_keep_failure_gate_reads_ramps_from_a_separate_folder(tmp_path):
    runs, types, rests, datums = _failure_population(tmp_path, with_ramp=False)
    ramps = tmp_path/"ramps"
    for cell in ("200", "300"):
        (ramps/f"transfer-all-{cell}-ramp").mkdir(parents=True)
        (ramps/f"transfer-all-{cell}-ramp"/"run.json").write_text(json.dumps(dict(cell=cell, **_ramp())))
    result = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums, ramp_folder=ramps)
    assert result["kept"] == ["200"] and result["pending"] == {} and result["ramp_folder"] == str(ramps)
    assert "transfer-all-200-ramp" in result["input_hashes"]
    without = decision.decide_keep(runs, types, rests, gate="failure", datums_path=datums)
    assert without["pending"] == {"200": "ramp_missing"}


def test_bands_gate_is_unchanged_by_the_failure_gate(tmp_path):
    runs, types, rests, _ = _failure_population(tmp_path)
    result = decision.decide_keep(runs, types, rests)
    assert result["gate"] == "bands" and result["kept"] == ["200"]   # 40 is outside the old band of 12
    assert "legacy_verdict" not in result["cells"]["200"] and "rule_counts" not in result


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


def test_cli_failure_gate_reports_pending_ramps_and_takes_a_ramp_folder(tmp_path, monkeypatch, capsys):
    runs, types, rests, datums = _failure_population(tmp_path, with_ramp=False)
    output = tmp_path/"decision.json"
    base = ["decide", "--folder", str(runs), "--output", str(output), "--keep", "--types", str(types),
            "--rests", str(rests), "--gate", "failure", "--datums", str(datums)]
    monkeypatch.setattr(sys, "argv", base)
    decision.main()
    out = capsys.readouterr().out
    assert "kept 0 / dropped 1 / missing 0 of 2; 1 pending (not yet measured)" in out
    assert "pending 200: ramp_missing" in out and "drop 300: count_in_type_spread" in out
    ramps = tmp_path/"ramps"
    (ramps/"transfer-all-200-ramp").mkdir(parents=True)
    (ramps/"transfer-all-200-ramp"/"run.json").write_text(json.dumps(dict(cell="200", **_ramp())))
    monkeypatch.setattr(sys, "argv", base+["--ramp-folder", str(ramps)])
    decision.main()
    assert json.loads(output.read_text())["kept"] == ["200"]


def test_ramp_readings_of_accepts_both_layouts_and_folds_finiteness():
    nested = _ramp(finite=True, pre=2)
    flat = decision.ramp_readings_of(nested)
    assert flat["rheobase_na"] == .055 and flat["pre_pulse_count"] == 2 and flat["finite"] is True
    nested["output_site"]["finite"] = False
    assert decision.ramp_readings_of(nested)["finite"] is False
    assert decision.ramp_readings_of(nested["ramp"]) is nested["ramp"] and decision.ramp_readings_of(None) is None
