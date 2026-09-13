"""Tests for the stage-13 scorer (the base as a sub-system)."""

import numpy as np
import pytest

import h01_topographic_donor as dn


def _train(thresholds, rises, start=1100., gap=60.):
    t = np.arange(0., 2100., .02)
    v = np.full_like(t, -70.)
    on = (t >= 1020.) & (t < 2020.)
    v[on] = -60.
    for i, (th, rise) in enumerate(zip(thresholds, rises)):
        s = start+i*gap
        seg = (t >= s) & (t < s+1.5)
        v[seg] = -60.+(40.+60.)*np.exp(-((t[seg]-s-.5)/(.16*640./rise))**2)
    return t, v


def test_train_rows_report_every_spike_in_order():
    t, v = _train([-55.]*6, [640.]*6)
    rows = dn.train_rows(t, v)
    assert [r["spike"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert all(np.isfinite(r["rise_v_s"]) and r["rise_v_s"] > 100. for r in rows)
    assert rows[1]["approach_mv_per_ms"] >= 0.


def test_climb_and_relation():
    rows = [{"spike": i+1, "threshold_mv": -56.+.5*i, "rise_v_s": 350.-5*i, "approach_mv_per_ms": .3} for i in range(6)]
    c = dn.climb(rows)
    assert c["climb_2_mv"] == pytest.approx(.5) and c["climb_5_mv"] == pytest.approx(2.) and c["rise_5_over_1"] == pytest.approx(330./350.)
    assert dn.climb(rows[:4]) is None
    runs = {"0.2": {"status": "read", "spikes": [dict(rows[0], approach_mv_per_ms=.2, threshold_mv=-54.5, span_mv=4.)]},
            "0.31": {"status": "read", "spikes": [dict(rows[0], approach_mv_per_ms=.7, threshold_mv=-56.7, span_mv=4.)]},
            "0.4": {"status": "read", "spikes": [dict(rows[0], approach_mv_per_ms=2.5, threshold_mv=-70., span_mv=19.)]}}
    rel = dn.take_off_relation(runs)
    assert rel["n"] == 2 and rel["slope_mv_per_mv_ms"] == pytest.approx(-4.4) and rel["excluded_out_of_band_spans"] == 1 and rel["overlaps_recorded_range"]


def _result(hl_rise, climb2, climb5, rise5, count310=10):
    def spikes(rise, c2, c5, r5, n):
        th = [-56., -56.+c2, -55., -54., -56.+c5] + [-52.]*(n-5)
        rs = [rise, rise*.95, rise*.92, rise*.9, rise*r5] + [rise*r5]*(n-5)
        return [{"spike": i+1, "threshold_mv": th[i], "take_off_mv": th[i], "rise_v_s": rs[i], "fall_v_s": -100., "peak_mv": 35., "approach_mv_per_ms": .5, "span_mv": 4., "t_ms": 1100.+60*i} for i in range(n)]
    hl = {d: {"status": "read", "count": count310, "spikes": spikes(hl_rise, climb2, climb5, rise5, max(count310, 5)), "rest_mv": -74.} for d in dn.RUNS}
    b3 = {d: {"status": "read", "count": 10, "spikes": spikes(570., 0., 0., 1.04, 10), "rest_mv": -84.} for d in dn.B3_RUNS}
    rec = {d: {"status": "read", "count": 10, "spikes": spikes(348., 1.4, 3.6, .86, 10), "rest_mv": -84.} for d in dn.RECORDED_SWEEPS}
    result = {"hl23pyr": hl, "b3": b3, "recorded": rec, "chain": {"tau_p_us": 5., "fc_khz": 10.}, "reading": ["r"]}
    result["take_off_relation"] = dn.take_off_relation(hl)
    return result


def test_decide_passes_a_human_like_base_and_fails_a_b3_like_one():
    d = dn.decide(_result(360., 1., 2.5, .85))
    assert d["verdict"] == "PASS"
    d = dn.decide(_result(600., 0., 0., 1.04))
    failed = [c["reading"] for c in d["checks"] if c["pass"] is False]
    assert d["verdict"] == "FAIL" and "a" in failed and "b" in failed
    d = dn.decide(_result(360., 1., 2.5, .85, count310=30))
    assert any(c["reading"] == "e" and c["pass"] is False for c in d["checks"])
    empty = {"hl23pyr": {d: {"status": "no spike", "count": 0} for d in dn.RUNS}, "b3": {}, "recorded": {}, "take_off_relation": None}
    assert dn.decide(empty)["verdict"] == "no reading"


def test_markdown_lists_systems_and_the_train():
    result = _result(360., 1., 2.5, .85)
    result["decision"] = dn.decide(result)
    md = dn.markdown(result)
    for piece in ("| 0.31 | HL23PYR | 10 |", "| 0.31 | B3 | 10 |", "| 0.31 | recording | 10 |", "## Threshold along the train", "Verdict: PASS"):
        assert piece in md
