"""Tests for the take-off against approach tables (SP15 stage 7)."""

import numpy as np
import pytest

import h01_topographic_threshold as thr


def _spike(t, v, onset, threshold, ramp_ms):
    """A linear approach over ``ramp_ms`` to ``threshold``, then a 1 ms spike and a trough."""
    ramp = (t >= onset-ramp_ms) & (t < onset)
    v[ramp] = -70.+(threshold+70.)*(t[ramp]-(onset-ramp_ms))/ramp_ms
    up = (t >= onset) & (t < onset+1.)
    v[up] = threshold+(40.-threshold)*np.sin(np.pi*(t[up]-onset))
    v[(t >= onset+1.) & (t < onset+4.)] = -80.


def _train(thresholds, ramps, onsets):
    t = np.arange(0., 3000., .02)
    v = np.full_like(t, -70.)
    for th, ramp, onset in zip(thresholds, ramps, onsets):
        _spike(t, v, onset, th, ramp)
    return t, v


def test_approach_rate_is_the_slope_and_stops_at_the_floor():
    t = np.arange(0., 100., .02)
    v = np.where(t < 50., -70., -70.+2.*(t-50.))   # 2 mV/ms after 50 ms
    assert thr.approach_rate(t, v, 60., floor_ms=0.) == pytest.approx(2., abs=1e-6)     # window 50..59.5, all on the ramp
    assert thr.approach_rate(t, v, 60., floor_ms=55.) == pytest.approx(2., abs=1e-6)    # floor inside the ramp
    assert np.isnan(thr.approach_rate(t, v, 52., floor_ms=51.6))                        # window shorter than 1 ms


def test_threshold_rows_read_a_faster_approach_and_use_the_previous_trough_as_floor():
    t, v = _train([-56., -54.], [20., 10.], [1100., 1200.])
    rows = thr.threshold_rows(t, v, (1020., 2020.))
    assert [r["spike"] for r in rows] == [1, 2]
    assert rows[0]["approach_mv_per_ms"] < rows[1]["approach_mv_per_ms"]
    assert rows[0]["approach_mv_per_ms"] == pytest.approx(14./20., rel=.15)
    assert rows[1]["approach_mv_per_ms"] == pytest.approx(16./10., rel=.15)
    assert rows[0]["threshold_mv"] == pytest.approx(-56., abs=1.)
    assert np.isfinite(rows[0]["crossing_mv"])


def test_first_crossing_is_not_applicable_when_the_approach_exceeds_the_criterion():
    t = np.arange(0., 100., .02)
    v = np.full_like(t, -70.)
    row = {"threshold_ms": 50., "peak_ms": 51., "approach_mv_per_ms": 7.}
    assert np.isnan(thr.first_crossing(t, v, row))


def test_relation_reports_the_slide_from_slowest_to_fastest_approach():
    tables = {"a": {"spikes": [{"spike": 1, "approach_mv_per_ms": .2, "threshold_mv": -55.}, {"spike": 2, "approach_mv_per_ms": .3, "threshold_mv": -53.}]},
              "b": {"spikes": [{"spike": 1, "approach_mv_per_ms": .7, "threshold_mv": -57.}]},
              "c": {"spikes": [{"spike": 1, "approach_mv_per_ms": float("nan"), "threshold_mv": -50.}]}}
    rel = thr.relation(tables)
    assert rel["n"] == 2
    assert rel["slide_mv"] == pytest.approx(-2.)
    assert thr.relation(tables, spike=None)["n"] == 3
    assert thr.relation({"a": {"spikes": []}}) is None


def test_later_spikes_residual_measures_against_the_spike_1_line():
    tables = {"a": {"spikes": [{"spike": 1, "approach_mv_per_ms": .2, "threshold_mv": -55.}, {"spike": 2, "approach_mv_per_ms": .2, "threshold_mv": -53.}]},
              "b": {"spikes": [{"spike": 1, "approach_mv_per_ms": .7, "threshold_mv": -57.}, {"spike": 2, "approach_mv_per_ms": .7, "threshold_mv": -58.}]}}
    res = thr.later_spikes_residual(tables, thr.relation(tables))
    assert res["n"] == 2
    assert res["median_residual_mv"] == pytest.approx(.5)
    assert res["min_residual_mv"] == pytest.approx(-1.)
    assert res["below_relation"] == 1
    assert thr.later_spikes_residual(tables, None) is None


def test_twins_contrast_and_input_resistance():
    table = {"56": {"count": 1, "steady": -67.0}, "57": {"count": 0, "steady": -66.8}, "59": {"count": 1, "steady": -66.6}}
    tc = thr.twins_contrast(table, .3)
    assert tc["diff_mv"] == pytest.approx(0.)
    assert tc["sigmas"] == pytest.approx(0.)
    stair = {"32": {"bias_pa": 2.5, "amplitude_pa": -100., "rest": -84., "steady": -92.},
             "33": {"bias_pa": 2.5, "amplitude_pa": -50., "rest": -84., "steady": -88.},
             "42": {"bias_pa": -3.7, "amplitude_pa": 100., "rest": -84., "steady": -75.},
             "43": {"bias_pa": -3.7, "amplitude_pa": 200., "rest": -84., "steady": -66.}}
    rin = thr.input_resistance(stair)
    assert rin["bias +2.5 pA"]["megaohm"] == pytest.approx(80.)
    assert rin["bias -3.7 pA"]["megaohm"] == pytest.approx(90.)


def test_decide_is_per_cell_and_fails_a_sliding_fit():
    def rel(slide, lo, hi):
        return {"threshold_mv": {"slide_mv": slide, "threshold_range": [lo, hi], "approach_range": [.2, 1.]},
                "crossing_mv": {"slide_mv": slide, "threshold_range": [lo, hi], "approach_range": [.2, 1.]}}
    result = {"relations": {"E human": rel(-2., -57., -55.), "E fit": rel(-.1, -57.2, -57.1),
                            "I human": rel(-4., -62., -58.), "I fit": rel(-.7, -61.1, -60.2)},
              "later_spikes": {"E human": {"median_residual_mv": 1.9}, "I human": {"median_residual_mv": 1.}},
              "twins_contrast": {"diff_mv": .16, "sigmas": .5}}
    d = thr.decide(result)
    assert d["verdict"] == {"E": "PASS", "I": "FAIL"}
    assert all(isinstance(c["pass"], bool) for c in d["checks"])
    assert len(d["checks"]) == 9


def test_markdown_carries_every_section():
    tab = {"amplitude_pa": 200., "bias_pa": -3.7, "spikes": [{"spike": 1, "approach_mv_per_ms": .2, "threshold_mv": -55., "crossing_mv": -55.5, "max_rise_v_s": 340., "t_ms": 60.}]}
    rel = {"threshold_mv": {"n": 1, "approach_range": [.2, .2], "threshold_range": [-55., -55.], "slide_mv": 0.}, "crossing_mv": {"slide_mv": 0.}}
    result = {"tables": {"E human": {"sweep 56": tab, "sweep 57": {"amplitude_pa": 200., "bias_pa": -3.7, "spikes": []}}},
              "relations": {"E human": rel}, "later_spikes": {"E human": None},
              "twins": {"sweep 56": {"count": 1, "rest": -84., "steady": -67., "post_100": -84., "post_300": -85., "post_1000": -84.}},
              "twins_contrast": {"diff_mv": .1, "sigmas": .3},
              "staircase": {"sweep 32": {"amplitude_pa": -110., "bias_pa": 2.5, "rest": -84., "steady": -92., "sag_mv": 1., "post_extreme_mv": -82.}},
              "input_resistance": {"bias +2.5 pA": {"megaohm": 75., "amplitude_range_pa": [-110., 70.], "n": 10}},
              "decision": {"checks": [{"cell": "E", "pass": True, "prediction": "p", "value_mv": 1., "sigmas": 4.}], "verdict": {"E": "PASS", "I": "FAIL"}},
              "reading": ["r"]}
    md = thr.markdown(result)
    for piece in ("no spike", "| 1 | 0.20 | -55.0 |", "## The 200 pA twins", "## Subthreshold staircase", "75.0 MOhm", "Verdict: E PASS, I FAIL", "- E PASS: p"):
        assert piece in md
