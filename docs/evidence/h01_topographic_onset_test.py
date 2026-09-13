"""Tests for the onset-shape tables (SP15 stage 8)."""

import numpy as np
import pytest

import h01_topographic_onset as ons


def _trace(onset_ms=1100., tau_ms=.05, lead_ms=0.):
    """One spike rising as an exponential in time: dV/dt grows exp(t/tau), so the 10 to 100 V/s span is 90*tau mV."""
    t = np.arange(0., 3000., .02)
    v = np.full_like(t, -70.)
    start = onset_ms-lead_ms
    ramp = (t >= start-20.) & (t < start)
    v[ramp] = -70.+10.*(t[ramp]-(start-20.))/20.
    up = (t >= start) & (t < start+8.*tau_ms)
    v[up] = -60.+.1*(np.exp((t[up]-start)/tau_ms)-1.)
    v[(t >= start+8.*tau_ms) & (t < start+8.*tau_ms+3.)] = -80.
    return t, v


def test_rate_of_rise_is_a_central_difference_over_0_1_ms():
    v = np.arange(0., 10., .02)*150.        # 3 mV per 0.02 ms sample = 150 V/s
    rate = ons.rate_of_rise(v)
    assert rate[10] == pytest.approx(150.)
    assert rate[0] == pytest.approx(150.) and rate[-1] == pytest.approx(150.)


def test_crossing_index_finds_the_first_reach_or_none():
    rate = np.array([0., 5., 12., 30., 120.])
    assert ons.crossing_index(rate, 0, 5, 10.) == 2
    assert ons.crossing_index(rate, 3, 5, 10.) == 3
    assert ons.crossing_index(rate, 0, 2, 10.) is None


def test_onset_span_grows_with_a_more_gradual_upstroke():
    sharp = ons.onset_rows(*_trace(tau_ms=.02), (1020., 2020.))[0]["soma"]
    soft = ons.onset_rows(*_trace(tau_ms=.06), (1020., 2020.))[0]["soma"]
    assert sharp["span_mv"] == pytest.approx(1.8, rel=.25)
    assert soft["span_mv"] > 2.5*sharp["span_mv"]          # the 0.1 ms difference smooths a 0.06 ms exponential; order, not value
    assert sharp["rapidness_per_ms"] > soft["rapidness_per_ms"] > 0.
    assert np.isfinite(sharp["low_ms"])


def test_axon_lead_is_positive_when_the_axon_fires_first():
    t, v = _trace()
    _, axon = _trace(lead_ms=.3)
    rows = ons.onset_rows(t, v, (1020., 2020.), axon)
    assert rows[0]["axon_lead_ms"] == pytest.approx(.3, abs=.03)
    assert rows[0]["axon_mv_at_soma_10"] > rows[0]["soma"]["low_mv"]
    lead = ons.axon_lead({"a": rows})
    assert lead["axon_first_every_spike"] and lead["n"] == 1 and len(lead["spike_1"]) == 1


def test_onset_is_nan_when_the_spike_never_reaches_100_v_s():
    t = np.arange(0., 200., .02)
    v = np.full_like(t, -70.)
    rate = np.full_like(t, 20.)
    row = {"threshold_ms": 100., "peak_ms": 101.}
    assert np.isnan(ons.onset(t, v, rate, row)["span_mv"])


def test_classify_and_summary_and_sigma():
    assert ons.classify(.5) == "kink" and ons.classify(1.5) == "unresolved" and ons.classify(3.) == "gradual"
    tables = {"a": [{"soma": {"span_mv": 2., "rapidness_per_ms": 30.}}, {"soma": {"span_mv": 3., "rapidness_per_ms": 20.}}],
              "b": [{"soma": {"span_mv": 4., "rapidness_per_ms": 40.}}], "c": []}
    summ = ons.summary(tables)
    assert summ["spike_1"]["span_mv"] == {"n": 2, "median": 3., "min": 2., "max": 4.}
    assert summ["later"]["span_mv"]["n"] == 1
    sig = ons.repeat_sigma(tables, {"a", "b"})
    assert sig["n"] == 2 and sig["span_range_mv"] == pytest.approx(2.)


def test_decide_refutes_a_gradual_fit_and_reads_both_classes():
    def summ(span, later=None):
        s = {"span_mv": {"n": 1, "median": span, "min": span, "max": span}, "rapidness_per_ms": {"n": 1, "median": 30., "min": 30., "max": 30.}}
        l = {"span_mv": None, "rapidness_per_ms": None} if later is None else {"span_mv": {"n": 1, "median": later, "min": later, "max": later}, "rapidness_per_ms": s["rapidness_per_ms"]}
        return {"spike_1": s, "later": l}
    result = {"summary": {"E human": {"soma": summ(5.)}, "E fit": {"soma": summ(5.1, 4.)}, "I human": {"soma": summ(2.7)}, "I fit": {"soma": summ(14.)}},
              "axon_lead": {"E fit": {"min_ms": .28, "axon_first_every_spike": True}, "I fit": {"min_ms": -.18, "axon_first_every_spike": False}},
              "sigma": {"E": {"span_mv": .26, "span_range_mv": .59}, "I": {"span_mv": .57, "span_range_mv": 1.16}}}
    d = ons.decide(result)
    assert d["classes"] == {"E": "recording gradual, fit gradual", "I": "recording gradual, fit gradual"}
    assert d["verdict"] == {"E": "FAIL", "I": "FAIL"}
    assert [c["pass"] for c in d["checks"] if c["cell"] == "E"] == [False, True, True, False, True]
    assert all(isinstance(c["pass"], bool) for c in d["checks"])


def test_markdown_lists_tables_with_and_without_an_axon_column():
    soma = {"span_mv": 5., "rapidness_per_ms": 30., "low_ms": 1100., "low_mv": -56.}
    result = {"tables": {"E human": {"sweep 48": [{"spike": 1, "threshold_mv": -55., "soma": soma}]},
                         "E fit": {"200 pA": [{"spike": 1, "threshold_mv": -57., "soma": soma, "axon": soma, "axon_lead_ms": .28, "axon_mv_at_soma_10": -40.}]}},
              "summary": {"E human": {"soma": ons.summary({"a": [{"soma": soma}]})}},
              "axon_lead": {"E fit": {"n": 1, "median_ms": .28, "min_ms": .28, "axon_first_every_spike": True, "axon_mv_at_soma_10_median": -40., "spike_1": [{"lead_ms": .28, "axon_mv_at_soma_10": -40.}]}},
              "sigma": {"E": {"n": 5, "span_mv": .26, "span_range_mv": .59, "rapidness_per_ms": 1.2}},
              "sp16_ais": {"depth 0": [{"spike": 1, "soma_10_mv": -57.2, "axon_10_mv": -55.1, "axon_lead_ms": .3}], "depth 0.6": None},
              "decision": {"checks": [{"cell": "E", "pass": False, "prediction": "p", "value": 5., "sigmas": .4}], "classes": {"E": "x", "I": "y"}, "verdict": {"E": "FAIL", "I": "FAIL"}},
              "reading": ["r"]}
    md = ons.markdown(result)
    for piece in ("| sweep 48 | 1 | -55.0 | 5.00 | 30 |", "| 200 pA | 1 | -57.0 | 5.00 | 30 | 5.00 | 0.280 | -40.0 |", "- E FAIL: p (+5.00, 0.4 sigma)", "Verdict: E FAIL, I FAIL", "| depth 0 | 1 | -57.2 | -55.1 | 0.30 |", "trace not retained"):
        assert piece in md
