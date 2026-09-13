"""Tests for the stage-10 scorer (soma-site coupling)."""

import pytest

import h01_topographic_coupling as cpl


def _ramp(status, approach, soma, axon0, axon1, span=4., lead=.3):
    if status != "read":
        return {"rate_pa_per_ms": 1., "status": status}
    return {"rate_pa_per_ms": 1., "status": "read", "count": 3,
            "spike_1": {"approach_mv_per_ms": approach, "points": {"soma": {"take_off_mv": soma, "span_mv": span, "lead_ms": 0.},
                                                                    "axon0": {"take_off_mv": axon0, "span_mv": 5., "lead_ms": lead/2},
                                                                    "axon1": {"take_off_mv": axon1, "span_mv": 6., "lead_ms": lead}}}}


def _arm(slide_soma, lead=.3, count=10, span=4.):
    ramps = {"ramp01": _ramp("read", .4, -57., -55.7, -54.7, span, lead), "ramp1": _ramp("read", 1., -57.+slide_soma/2, -56.5, -55.8, span, lead),
             "ramp10": _ramp("read", 2.5, -57.+slide_soma, -58.1, -58.8, span, lead)}
    arm = {"ramps": ramps, "step": {"status": "read", "count": count, "cycles": []}}
    arm["summary"] = cpl.summarise(arm)
    return arm


def test_slide_orders_by_approach_and_needs_two_ramps():
    ramps = {"b": _ramp("read", 2.5, -58., -58.1, -58.8), "a": _ramp("read", .4, -57., -55.7, -54.7), "c": _ramp("no spike", 0, 0, 0, 0)}
    s = cpl.slide(ramps, "soma")
    assert s["n"] == 2 and s["slide_mv"] == pytest.approx(-1.) and s["approach_range"] == [.4, 2.5]
    assert cpl.slide({"a": ramps["a"], "c": ramps["c"]}, "soma") is None


def test_summarise_reports_spans_leads_and_count():
    arm = _arm(-1.2, lead=.1, count=9)
    su = arm["summary"]
    assert su["soma"]["slide_mv"] == pytest.approx(-1.2)
    assert su["axon1"]["slide_mv"] == pytest.approx(-4.1)
    assert su["soma_span_range_mv"] == [4., 4.] and su["axon1_lead_range_ms"] == [.1, .1] and su["count_310"] == 9


def test_decide_passes_the_registered_predictions_and_fails_a_fixed_soma():
    result = {"arms": {"control 1 um": _arm(0.), "x2 total held": _arm(-.7), "x2 density held": _arm(-.8), "x4 total held": _arm(-1.3, lead=.1)}}
    d = cpl.decide(result)
    assert d["verdict"] == "PASS"
    result["arms"]["x4 total held"] = _arm(0., lead=.1)
    assert cpl.decide(result)["verdict"] == "FAIL"
    result["arms"]["x4 total held"]["ramps"]["ramp10"] = {"rate_pa_per_ms": 110., "status": "no spike"}
    result["arms"]["x4 total held"]["ramps"]["ramp1"] = {"rate_pa_per_ms": 11., "status": "not run"}
    result["arms"]["x4 total held"]["summary"] = cpl.summarise(result["arms"]["x4 total held"])
    assert cpl.decide(result)["verdict"] == "no reading"


def test_decide_flags_disagreeing_x2_arms_and_a_changed_spike():
    result = {"arms": {"control 1 um": _arm(0.), "x2 total held": _arm(-.6), "x2 density held": _arm(-1.5), "x4 total held": _arm(-1.3, lead=.1, span=6.5)}}
    d = cpl.decide(result)
    failed = [c["prediction"] for c in d["checks"] if c["pass"] is False]
    assert any("agree" in p for p in failed) and any("span" in p for p in failed)


def test_markdown_lists_arms_and_decision():
    result = {"arms": {name: _arm(-1.) for name in cpl.ARMS}, "reading": ["r"]}
    result["arms"]["x4 total held"]["step"] = {"status": "not registered"}
    result["decision"] = cpl.decide(result)
    md = cpl.markdown(result)
    for piece in ("## x4 total held (diameter 4.0 um, NaTs axon 0.9535 S/cm2)", "| 1.0 | read | 3 | 0.40 | -57.0 | -55.7 | -54.7 | 4.0 | +0.30 |", "310 pA step: 10 spikes", "310 pA step: not registered", "Verdict:"):
        assert piece in md
