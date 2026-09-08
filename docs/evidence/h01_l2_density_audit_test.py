"""Test direct-event decisions, including lost and malformed events."""

import pytest

from docs.evidence.h01_l2_density_audit import assess_events


def events(start, interval, peak):
    return [dict(rise_crossing_ms=start + i * interval, peak_sample_voltage_mv=peak,
                 time_above_threshold_ms=1.) for i in range(5)]


@pytest.mark.parametrize("case,decision", [
    ("improve", "supported"), ("onset", "rejected"), ("peak", "rejected"),
    ("interval", "rejected"), ("lost", "rejected"), ("extra", "rejected"),
    ("none", "invalid"), ("nan", "invalid"), ("order", "invalid"),
    ("human", "invalid"), ("first_peak", "invalid"),
])
def test_direct_predictions(case, decision):
    control, candidate, human = events(1090, 90, 40), events(1095, 95, 37), events(1100, 100, 35)
    if case == "onset":
        candidate[0]["rise_crossing_ms"] = 1080
    elif case == "peak":
        candidate[0]["peak_sample_voltage_mv"] = 41
    elif case == "interval":
        candidate[1]["rise_crossing_ms"] -= 30
    elif case == "lost":
        candidate.pop()
    elif case == "extra":
        candidate.append(dict(candidate[-1], rise_crossing_ms=1600))
    elif case == "none":
        candidate = []
    elif case == "nan":
        candidate[0]["rise_crossing_ms"] = float("nan")
    elif case == "order":
        candidate.reverse()
    elif case == "human":
        human.pop()
    elif case == "first_peak":
        candidate[0]["peak_sample_voltage_mv"] = -1
    result = assess_events(control, candidate, human)
    assert result["decision"] == decision
    if case == "lost":
        assert len(result["records"][1]["unmatched_human_events"]) == 1
    if case == "interval":
        assert not result["checks"]["no_interval_error_worse"]
