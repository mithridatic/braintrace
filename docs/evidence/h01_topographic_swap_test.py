"""Tests for the stage-15 scorer (the human-for-rodent sodium swap and the percent accuracy)."""

import numpy as np
import pytest

import h01_topographic_swap as sw


def _run(rise, count=10, fall=-100., peak=30., thr=-55., take_off=-55., rest=-84., c2=1.3, c5=3.4, r5=.86):
    spikes = []
    n = max(count, 5)
    for i in range(n):
        t = thr + (c2 if i == 1 else c5 if i == 4 else 0.)
        spikes.append({"spike": i+1, "threshold_mv": t, "take_off_mv": take_off, "span_mv": 4.,
                       "rise_v_s": rise*(r5 if i == 4 else 1.), "fall_v_s": fall, "peak_mv": peak,
                       "approach_mv_per_ms": .5, "t_ms": 1100.+60*i})
    return {"status": "read", "count": count, "spikes": spikes, "rest_mv": rest}


def test_curve_sorts_by_dose_and_skips_unread():
    runs = {"0.272": _run(500.), "0.068": _run(300.), "0.544": {"status": "no spike", "count": 0}}
    pts = sw.curve(runs)
    assert [p["dose_s_cm2"] for p in pts] == [0.068, 0.272]
    assert pts[0]["rise_v_s"] == 300.


def test_crossing_interpolates_inside_and_reports_outside():
    pts = [{"dose_s_cm2": .1, "rise_v_s": 300., "count": 8}, {"dose_s_cm2": .2, "rise_v_s": 400., "count": 9}]
    c = sw.crossing(pts, 350.)
    assert c["inside_series"] and c["dose_s_cm2"] == pytest.approx(.15)
    assert sw.crossing(pts, 900.)["inside_series"] is False


def test_accuracy_is_perfect_on_an_identical_run_and_uses_the_stated_scales():
    rec = _run(348.)
    a = sw.accuracy(_run(348.), rec)
    assert a["mean_pct"] == pytest.approx(100.)
    assert a["n_elements"] == 10
    # a 34.8 V/s rise error is 10 percent of the recorded rise
    off = sw.accuracy(_run(348.*1.1), rec)
    assert off["elements"]["rise_v_s"]["accuracy_pct"] == pytest.approx(90., abs=.5)
    # a 10 mV level error is scored against the stated 100 mV span
    lvl = sw.accuracy(_run(348., rest=-74.), rec)
    assert lvl["elements"]["rest_mv"]["accuracy_pct"] == pytest.approx(90., abs=.5)


def test_accuracy_floors_at_zero_and_handles_an_unread_run():
    rec = _run(348.)
    bad = sw.accuracy(_run(348., c2=-40.), rec)
    assert bad["elements"]["climb_2_mv"]["accuracy_pct"] == 0.
    assert sw.accuracy({"status": "no spike", "spikes": []}, rec) is None


def _result(rises, counts=(10, 10, 10, 10)):
    rec = _run(348.)
    runs = {d: _run(r, count=c) for d, r, c in zip(sw.DOSES, rises, counts)}
    pts = sw.curve(runs)
    acc = {"b3": sw.accuracy(_run(570.), rec)}
    best, best_dose = None, None
    for d, r in runs.items():
        a = sw.accuracy(r, rec)
        if best is None or a["mean_pct"] > best["mean_pct"]:
            best, best_dose = a, d
    acc["best_human"] = best
    return {"curve": pts, "crossing": sw.crossing(pts, 348.), "recorded_reference": {"rise_v_s": 348., "count": 10},
            "b3": _run(570.), "accuracy": acc, "best_dose": best_dose, "runs": runs, "recorded": rec}


def test_decide_passes_when_the_curve_is_monotone_and_a_dose_lands_in_band_with_the_count():
    d = sw.decide(_result([300., 348., 500., 700.]))
    assert d["verdict"] == "PASS"
    assert next(c for c in d["checks"] if c["reading"] == "a")["pass"] is True
    assert next(c for c in d["checks"] if c["reading"] == "b")["pass"] is True


def test_decide_fails_cell_b_when_the_in_band_dose_breaks_the_count():
    d = sw.decide(_result([300., 348., 500., 700.], counts=(2, 40, 10, 10)))
    assert next(c for c in d["checks"] if c["reading"] == "b")["pass"] is False
    assert d["verdict"] == "MIXED"


def test_decide_flags_a_non_monotone_curve():
    d = sw.decide(_result([500., 300., 600., 700.]))
    assert next(c for c in d["checks"] if c["reading"] == "a")["pass"] is False


def test_decide_reports_no_reading_without_two_doses():
    assert sw.decide({"curve": [], "crossing": None, "recorded_reference": {"rise_v_s": 348.},
                      "b3": {"status": "no spike"}, "accuracy": {"b3": None}})["verdict"] == "no reading"


def test_arm_b_is_recorded_as_a_no_reading_with_its_cause():
    assert sw.ARM_B["status"] == "no reading"
    assert "unchanged condition" in sw.ARM_B["cause"]
