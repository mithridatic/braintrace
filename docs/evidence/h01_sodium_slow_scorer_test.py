"""Tests for the SP16 per-cycle scorer."""

import pytest

import h01_sodium_slow_scorer as sc

LIMITS = {"threshold_mv": 1., "rise_fraction": .035, "first_spike_ms": 23.}


def _table(thresholds, rise=640., t0=34., interval=100.):
    spikes = [{"spike": i+1, "threshold_mv": th, "max_rise_v_s": rise if i else rise, "t_ms": t0+i*interval} for i, th in enumerate(thresholds)]
    first = spikes[0]
    shape = None
    if len(spikes) > 1:
        shape = {"spike_2": {"threshold_mv": spikes[1]["threshold_mv"]-first["threshold_mv"], "rise_fraction": 1., "t_ms": interval},
                 "last": {"threshold_mv": spikes[-1]["threshold_mv"]-first["threshold_mv"], "rise_fraction": 1., "t_ms": interval*(len(spikes)-1)}}
    return {"count": len(spikes), "spikes": spikes, "shape": shape}


def test_reproduction_passes_an_identical_run_and_fails_a_moved_threshold():
    ref = _table([-57.2]*10)
    assert all(r["pass"] for r in sc.reproduction(ref, ref, LIMITS))
    moved = _table([-57.2]*9+[-57.0])
    rows = sc.reproduction(moved, ref, LIMITS)
    assert [r["check"] for r in rows if not r["pass"]] == ["every threshold within 0.1 mV"]


def test_reproduction_fails_a_count_change_without_crashing_on_an_empty_run():
    ref = _table([-57.2]*10)
    rows = sc.reproduction({"count": 0, "spikes": [], "shape": None}, ref, LIMITS)
    assert rows == [{"check": "count equal", "pass": False, "run": 0, "reference": 10}]


def test_dose_310_reads_the_step_and_the_hold():
    ref = _table([-57.2]*10)
    stepped = _table([-57.2, -55.6, -55.0, -54.6, -54.2, -54.1, -54.2, -54.1, -54.2, -54.1])
    rows = {r["check"]: r for r in sc.dose_310(stepped, ref, LIMITS)}
    assert rows["threshold step at spike 2 above the limit"]["pass"] and rows["threshold step at spike 2 above the limit"]["run"] == pytest.approx(1.6)
    assert rows["hold from spike 5 (spike 10 minus spike 5 within the limit)"]["pass"]
    assert rows["spike-1 rise unchanged"]["pass"] and rows["count 9 or 10"]["pass"]
    accumulating = _table([-57.2, -55.6, -55.0, -54.6, -54.2, -53.6, -53.2, -52.8, -52.4, -52.0])
    rows = {r["check"]: r for r in sc.dose_310(accumulating, ref, LIMITS)}
    assert not rows["hold from spike 5 (spike 10 minus spike 5 within the limit)"]["pass"]


def test_dose_310_fails_when_spike_1_moves_or_the_count_drops():
    ref = _table([-57.2]*10)
    run = _table([-57.2, -55.6, -55.0, -54.6, -54.2, -54.1, -54.2, -54.1])
    run["spikes"][0]["max_rise_v_s"] = 600.
    rows = {r["check"]: r for r in sc.dose_310(run, ref, LIMITS)}
    assert not rows["count 9 or 10"]["pass"] and not rows["spike-1 rise unchanged"]["pass"]
    assert "hold from spike 5 (spike 10 minus spike 5 within the limit)" not in rows


def test_dose_200_reads_the_count_and_the_level():
    ref = _table([-57.2]*4, t0=127.)
    ref["level_p50_mv"] = -65.
    run = _table([-57.2], t0=130.)
    run["level_p50_mv"] = -66.5
    rows = {r["check"]: r for r in sc.dose_200(run, ref, LIMITS)}
    assert all(r["pass"] for r in rows.values())
    run["level_p50_mv"] = -64.
    rows = {r["check"]: r for r in sc.dose_200(run, ref, LIMITS)}
    assert not rows["post-spike level towards -67.1 (reference -65.0)"]["pass"]
