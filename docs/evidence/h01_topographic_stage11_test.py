"""Tests for the stage-11 scorer (rise dose; what a spike leaves behind)."""

import numpy as np
import pytest

import h01_topographic_stage11 as s11


def _threshold_result():
    def sp(i, a, v, t):
        return {"spike": i, "approach_mv_per_ms": a, "threshold_mv": v, "t_ms": t, "max_rise_v_s": 340.}
    long = {"sweep 53": {"spikes": [sp(1, .3, -55., 50.), sp(2, .2, -53., 80.), sp(3, .2, -53., 120.), sp(4, .2, -53.2, 160.),
                                    sp(5, .2, -52.9, 200.), sp(6, .2, -53.1, 240.), sp(7, .2, -52.8, 280.)]},
            "sweep 50": {"spikes": [sp(1, .4, -55.5, 60.), sp(2, .2, -53.2, 90.)]}}
    short = {f"short sweep {s}": {"spikes": [sp(1, 7.2, v, 3.3)]} for s, v in zip(s11.E_SHORT, (-61.9, -62.0, -61.3, -61.6, -61.7))}
    return {"relations": {"E human": {"threshold_mv": {"slope_mv_per_mv_ms": -5.}}}, "tables": {"E human": {**long, **short}}}


def test_take_off_move_corrects_by_the_ramp_slope():
    ref = {"take_off_mv": -55.3, "approach_mv_per_ms": .4}
    row = {"take_off_mv": -56.0, "approach_mv_per_ms": 4.}
    mv = s11.take_off_move(row, ref, slope=-2.8)
    assert mv["raw_mv"] == pytest.approx(-.7) and mv["approach_decades"] == pytest.approx(1.)
    assert mv["corrected_mv"] == pytest.approx(2.1)


def test_short_square_rows_measure_against_the_unprimed_sweep():
    timing = {27: 26., 28: 4.15, 29: 4.18, 30: 2.72, 31: 15.8}
    ss = s11.short_square_rows(_threshold_result()["tables"]["E human"], timing)
    assert [r["minus_unprimed_mv"] for r in ss["rows"]] == pytest.approx([0., -.1, .6, .3, .2])
    assert ss["primed_seconds_range"] == [2.72, 15.8]
    assert ss["sigma_mv"] == pytest.approx(np.std([-61.9, -62., -61.3, -61.6, -61.7], ddof=1))


def test_residual_by_spike_bands_and_growth():
    rb = s11.residual_by_spike(_threshold_result())
    assert rb["n"] == 7
    assert rb["bands"]["spike_2"]["n"] == 2 and rb["bands"]["spikes_6_plus"]["n"] == 2
    assert rb["growth_mv"] == pytest.approx(rb["bands"]["spikes_6_plus"]["median_mv"]-rb["bands"]["spike_2"]["median_mv"])
    assert rb["slope_mv_per_spike"] is not None


def _dose(status, rise=600., take_off=-55.3, approach=.4, count=9):
    if status != "read":
        return {"status": status}
    return {"status": "read", "count": count, "rise_v_s": rise, "take_off_mv": take_off, "approach_mv_per_ms": approach, "span_mv": 4.5}


def _result(d07, d05):
    doses = {"0.9": _dose("read", 655.), "0.7": d07, "0.5": d05}
    for dose in ("0.7", "0.5"):
        if doses[dose]["status"] == "read":
            doses[dose]["move"] = s11.take_off_move(doses[dose], doses["0.9"])
    return {"rise": {"doses": doses, "recorded_rise_v_s": 348.},
            "short_squares": s11.short_square_rows(_threshold_result()["tables"]["E human"], {27: 26., 28: 4.15, 29: 4.18, 30: 2.72, 31: 15.8}),
            "residual_by_spike": s11.residual_by_spike(_threshold_result()), "reading": ["r"]}


def test_decide_passes_a_monotone_series_with_the_take_off_held_and_fails_a_moved_one():
    d = s11.decide(_result(_dose("read", 520., -55.2, .5), _dose("read", 400., -55.1, .45)))
    assert d["verdict"] == "PASS"
    d = s11.decide(_result(_dose("read", 520., -55.2, .5), _dose("read", 400., -53.9, .6, count=7)))
    failed = [c["prediction"] for c in d["checks"] if c["pass"] is False]
    assert d["verdict"] == "FAIL" and any("0.5, corrected" in p for p in failed) and any("count at 0.5" in p for p in failed)
    d = s11.decide(_result(_dose("read", 700., -55.2, .5), _dose("read", 400., -55.1, .45)))
    assert any("monotonically" in c["prediction"] and c["pass"] is False for c in d["checks"])
    d = s11.decide(_result(_dose("not run"), _dose("no spike")))
    assert d["verdict"] == "no reading"


def test_markdown_lists_doses_short_squares_bands_and_decision():
    result = _result(_dose("read", 520., -55.2, .5), _dose("not run"))
    result["decision"] = s11.decide(result)
    md = s11.markdown(result)
    for piece in ("| 0.9 | read | 9 | 655 | -55.30 | 0.40 | 4.5 | ref | ref |", "| 0.7 | read | 9 | 520 |", "| 0.5 | not run |", "| 27 | 26.0 | 7.20 | -61.90 | +0.00 |",
                  "| spike_2 | 2 |", "Verdict: no reading", "Recorded spike-1 rise (sweep 53): 348 V/s."):
        assert piece in md
