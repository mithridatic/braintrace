"""Residual vectors, observation selection, family ranking, and preservation logic."""
import numpy as np
import pytest
from docs.evidence import h01_l2_campaign_score as score

T = np.arange(0., 2100.001, .02)


def _trace(centres, shift=0., depth=-70.):
    v = np.full_like(T, -80.)
    for c in centres:
        v += 115.*np.exp(-((T-c-shift)/.4)**2)-(depth+80.)*np.exp(-((T-c-shift-2.)/1.5)**2)
    return v


HUMAN_CENTRES = [1078., 1112., 1332., 1640., 1913.]
LIMITS = {"fields": {"rise_crossing_ms": .0357, "peak_sample_voltage_mv": .01, "time_above_threshold_ms": .0004},
          "phase_ms": .0004, "minimum_voltage_mv": .0013}


def _datums():
    v = _trace(HUMAN_CENTRES)
    mask = (T >= 1020.) & (T < 2020.)
    events = score.spike_datums(T[mask], v[mask])
    recovery = []
    for i, e in enumerate(events):
        stop = events[i+1]["rise_crossing_ms"] if i+1 < len(events) else 2020.
        window = (T > e["fall_crossing_ms"]) & (T < stop)
        at = int(np.argmin(v[window]))
        recovery.append({"event_index": i, "minimum_sample_voltage_mv": float(v[window][at]),
                         "delay_from_peak_ms": float(T[window][at])-e["peak_sample_time_ms"]})
    return events, recovery


def test_identical_trace_gives_zero_residuals_and_shift_gives_interval_errors():
    events, recovery = _datums()
    same = score.active_residuals(T, _trace(HUMAN_CENTRES), events, recovery)
    assert same["event_count_model"] == 5 and all(abs(same[k]) < 1e-9 for k in same if not k.startswith("event_count"))
    late = score.active_residuals(T, _trace([1078., 1112.+3., 1332.+3., 1640.+3., 1913.+3.]), events, recovery)
    assert late["i1"] == pytest.approx(3., abs=1e-6) and late["i2"] == pytest.approx(0., abs=1e-6)
    assert late["e2_rise"] == pytest.approx(3., abs=1e-6) and abs(late["m1_v"]) < 1e-9
    missing = score.active_residuals(T, _trace(HUMAN_CENTRES[:3]), events, recovery)
    assert missing["event_count_model"] == 3 and np.isnan(missing["e4_rise"]) and np.isnan(missing["m4_v"]) and np.isnan(missing["i3"])


def test_subthreshold_residuals_interpolate_at_requested_times():
    samples = [{"requested_time_ms": 1019., "voltage_mv": -84.5}, {"requested_time_ms": 1040., "voltage_mv": -78.5}]
    out = score.subthreshold_residuals(np.array([1000., 1050.]), np.array([-84.5, -84.5]), samples)
    assert out == {"sub_1019_v": 0., "sub_1040_v": -6.}


def test_selection_and_ranking_name_the_dominant_family():
    source = {"i1": 0., "i2": 0., "e1_peak_v": 0., "event_count_model": 5}
    candidate = {"i1": 10., "i2": 0.01, "e1_peak_v": 1., "event_count_model": 5}
    keys = score.select_observations(source, candidate, LIMITS)
    assert keys == ["i1", "e1_peak_v"]
    vectors = {"source": source, "candidate": candidate,
               "into-F1": {"i1": 9.5, "e1_peak_v": .9}, "out-F1": {"i1": .5, "e1_peak_v": .1},
               "into-F2": {"i1": .2, "e1_peak_v": .05}, "out-F2": {"i1": 9.8, "e1_peak_v": .95}}
    ranking = score.rank_families(vectors, ["F1", "F2"], keys, LIMITS)
    assert ranking["order"] == ["F1", "F2"] and ranking["steep_x"] == "F1"
    tree = score.tree_markdown(ranking)
    assert "F1[F1: RSS" in tree and "Steep X" in tree and tree.startswith("flowchart TD")


def test_preservation_detects_sign_ordering_and_size_failures():
    fine = {"source": {"i1": 10.}, "candidate": {"i1": -4.}, "corner": {"i1": 2.}}
    good = {"source": {"i1": 10.1}, "candidate": {"i1": -4.1}, "corner": {"i1": 2.05}}
    assert score.preservation(good, fine, ["i1"], LIMITS)["preserved"]
    flipped = {"source": {"i1": 10.}, "candidate": {"i1": 4.}, "corner": {"i1": 2.}}
    assert ("i1", "sign") in score.preservation(flipped, fine, ["i1"], LIMITS)["failures"]
    swapped = {"source": {"i1": 1.}, "candidate": {"i1": -4.}, "corner": {"i1": 2.}}
    assert ("i1", "ordering") in score.preservation(swapped, fine, ["i1"], LIMITS)["failures"]
    large = {"source": {"i1": 12.}, "candidate": {"i1": -4.}, "corner": {"i1": 2.}}
    assert any("one fifth" in f[1] for f in score.preservation(large, fine, ["i1"], LIMITS)["failures"])


def test_limits_mark_proxies():
    assert score.limit_for("m1_v", LIMITS) == (.0013, "measured")
    assert score.limit_for("sub_1019_v", LIMITS) == (.0013, "proxy")
    assert score.limit_for("m1_delay", LIMITS) == (.0357, "proxy")
    assert score.limit_for("e1_rise_to_peak", LIMITS) == (.0004, "measured")
