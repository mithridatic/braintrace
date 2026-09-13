"""Tests for the stage-9 scorer (the take-off's own variables)."""

import numpy as np
import pytest

import h01_topographic_stage9 as s9


def _threshold_result():
    spikes_a = [{"spike": 1, "approach_mv_per_ms": .2, "threshold_mv": -55., "t_ms": 50.},
                {"spike": 2, "approach_mv_per_ms": .2, "threshold_mv": -53., "t_ms": 70.},
                {"spike": 3, "approach_mv_per_ms": .2, "threshold_mv": -53.5, "t_ms": 270.}]
    spikes_b = [{"spike": 1, "approach_mv_per_ms": .7, "threshold_mv": -57., "t_ms": 30.},
                {"spike": 2, "approach_mv_per_ms": .7, "threshold_mv": -55., "t_ms": 40.},
                {"spike": 3, "approach_mv_per_ms": float("nan"), "threshold_mv": -55., "t_ms": 60.}]
    return {"relations": {"E human": {"threshold_mv": {"slope_mv_per_mv_ms": -4.}}},
            "tables": {"E human": {"a": {"spikes": spikes_a}, "b": {"spikes": spikes_b}}}}


def test_residual_by_interval_skips_nan_approach_and_measures_the_interval():
    rows = s9.residual_by_interval(_threshold_result())
    assert [(r["train"], r["spike"]) for r in rows] == [("a", 2), ("a", 3), ("b", 2)]
    assert rows[0]["interval_ms"] == pytest.approx(20.)
    assert rows[1]["interval_ms"] == pytest.approx(200.)
    assert rows[0]["residual_mv"] == pytest.approx(2.)          # spike-1 line passes through both spike-1 points exactly
    assert rows[2]["residual_mv"] == pytest.approx(2.)


def test_interval_relation_bands_and_slope():
    rows = [{"interval_ms": 10., "residual_mv": 2.}, {"interval_ms": 20., "residual_mv": 2.}, {"interval_ms": 50., "residual_mv": 1.5},
            {"interval_ms": 200., "residual_mv": 1.}, {"interval_ms": 400., "residual_mv": 1.}]
    rel = s9.interval_relation(rows)
    assert rel["n"] == 5
    assert rel["bands"]["short_under_30_ms"]["median_mv"] == pytest.approx(2.)
    assert rel["bands"]["long_over_100_ms"]["median_mv"] == pytest.approx(1.)
    assert rel["band_spread_mv"] == pytest.approx(1.)
    assert rel["slope_mv_per_decade"] < 0.
    assert s9.interval_relation(rows[:2]) is None


def _spike_traces(lead_ms):
    t = np.arange(0., 3000., .02)
    def one(shift):
        v = np.full_like(t, -70.)
        start = 1100.-shift
        ramp = (t >= start-20.) & (t < start)
        v[ramp] = -70.+10.*(t[ramp]-(start-20.))/20.
        up = (t >= start) & (t < start+.4)
        v[up] = -60.+.1*(np.exp((t[up]-start)/.05)-1.)
        v[(t >= start+.4) & (t < start+3.4)] = -80.
        return v
    return {"soma": (t, one(0.)), "axon0": (t, one(lead_ms)), "axon1": (t, one(2*lead_ms))}


def test_point_rows_read_leads_and_take_offs_at_every_point():
    rows = s9.point_rows(_spike_traces(.1), (1020., 2020.), 1020.)
    assert len(rows) == 1
    p = rows[0]["points"]
    assert p["soma"]["lead_ms"] == pytest.approx(0.)
    assert p["axon0"]["lead_ms"] == pytest.approx(.1, abs=.03)
    assert p["axon1"]["lead_ms"] == pytest.approx(.2, abs=.03)
    assert np.isfinite(p["soma"]["take_off_mv"]) and np.isfinite(p["soma"]["span_mv"])


def test_e_ramp_contrast_needs_two_read_ramps_and_reports_ranges():
    def run(status, soma, axon1):
        return {"rate_pa_per_ms": 1., "status": status, "spike_1": {"approach_mv_per_ms": 1., "points": {"soma": {"take_off_mv": soma}, "axon1": {"take_off_mv": axon1}}} if status == "read" else None}
    assert s9.e_ramp_contrast({"a": run("read", -57., -55.), "b": run("no spike", 0., 0.)}) is None
    c = s9.e_ramp_contrast({"a": run("read", -57., -55.), "b": run("read", -57.4, -54.)})
    assert c["soma"]["range_mv"] == pytest.approx(.4)
    assert c["axon1"]["range_mv"] == pytest.approx(1.)
    assert "axon0" not in c


def test_decide_reads_fixed_step_no_ramp_reading_and_probe_pass_with_rejection():
    result = {"e_interval": {"relation": {"band_spread_mv": .69, "slope_mv_per_decade": -.57}},
              "e_ramps": {"runs": {"ramp01": {"status": "not run"}}, "contrast": None},
              "i_probes": {"status": "read", "max_lead_ms": {"axon0_0.5": .06, "axon1_0.5": .08}, "min_span_mv": {"soma": 12.8, "axon1_0.5": 8.9},
                           "earliest_point_spike_1": "axon1_0.5", "earliest_lead_spike_1_ms": .08}}
    d = s9.decide(result)
    a, b, c = d["checks"]
    assert a["fixed"] and not a["recovering"]
    assert b["pass"] is None and "not run" in b["note"]
    assert c["pass"] and c["rejection"].startswith("fired by the letter")
    result["i_probes"]["max_lead_ms"]["axon1_0.5"] = .3
    assert s9.decide(result)["checks"][2]["pass"] is False


def test_markdown_renders_every_section_without_ramps():
    result = {"e_interval": {"rows": [{"train": "sweep 49", "spike": 2, "interval_ms": 20., "approach_mv_per_ms": .2, "residual_mv": 2.}],
                             "relation": {"n": 1, "interval_range_ms": [20., 20.], "slope_mv_per_decade": 0., "bands": {"short_under_30_ms": {"n": 1, "median_mv": 2.}, "middle": None, "long_over_100_ms": None}, "band_spread_mv": 0.}},
              "e_ramps": {"runs": {"ramp01": {"rate_pa_per_ms": 1.1, "status": "not run"}}, "contrast": None},
              "i_probes": {"status": "not run"},
              "decision": {"checks": [{"cell": "E", "reading": "a", "prediction": "p", "value": 0., "fixed": True},
                                      {"cell": "E", "reading": "b", "prediction": "q", "value": None, "pass": None, "note": "none"}]},
              "reading": ["r"]}
    md = s9.markdown(result)
    for piece in ("| sweep 49 | 2 | 20 | 0.20 | +2.00 |", "| 1.1 | not run |", "not run", "- E (a) fixed: p (+0.00)", "- E (b) no reading: q (n/a; none)"):
        assert piece in md
