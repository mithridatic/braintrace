"""Band rows, rejection clause and tier verdicts of the I reserve stage-1 scorer (synthetic tables)."""

import pytest

from docs.evidence.h01_i_reserve_bands import band_rows, decide, rejection, tier_verdicts


def _table(cycles, troughs, width=.221, peak=17.9):
    return [{"cycle": i, "cycle_ms": c, "ahp_mv": t, "width_ms": width, "peak_mv": peak}
            for i, (c, t) in enumerate(zip(cycles, troughs), start=1)]


def _finalist_like():
    return {"0.19 nA": _table([31.2, 34.8, 36.] + [60.]*11, [-80.1, -80.5, -80.7] + [-80.8]*11),
            "0.27 nA": _table([13.6, 16.4, 17.] + [25.]*34, [-79., -79.3, -79.6] + [-79.8]*34)}


def test_band_rows_hold_when_every_registered_row_is_inside():
    table = _table([13., 11.5, 12.] + [22.]*38, [-79.2, -79.5, -79.9] + [-80.]*38)
    rows = band_rows("0.27 nA", table)
    assert [r["row"] for r in rows] == ["cycle2_ms", "count", "width1_ms", "peak_mv", "trough1_mv", "trough2_mv", "trough3_mv"]
    assert all(r["held"] for r in rows) and rows[1]["observed"] == 41


def test_band_rows_miss_on_the_unchanged_finalist():
    rows = {r["row"]: r for r in band_rows("0.27 nA", _finalist_like()["0.27 nA"])}
    assert not rows["cycle2_ms"]["held"] and not rows["count"]["held"]
    assert rows["width1_ms"]["held"] and rows["peak_mv"]["held"] and rows["trough1_mv"]["held"]


def test_band_rows_report_missing_cycles_as_missed():
    rows = {r["row"]: r for r in band_rows("0.19 nA", _table([30.], [-80.]))}
    assert rows["cycle2_ms"]["observed"] is None and not rows["cycle2_ms"]["held"]
    assert rows["trough3_mv"]["observed"] is None and not rows["trough3_mv"]["held"]


def test_rejection_when_cycle2_moves_less_than_the_limit():
    reject = rejection(_finalist_like())
    assert reject["cycle2_027_change_ms"] == pytest.approx(16.37-16.4)
    assert reject["change_below_limit"] and reject["met"] and reject["loop_breaks"] == []


def test_rejection_not_met_when_cycle2_moves_past_the_limit_and_the_loop_holds():
    tables = _finalist_like()
    tables["0.27 nA"][1]["cycle_ms"] = 3.
    reject = rejection(tables)
    assert not reject["change_below_limit"] and not reject["met"]


def test_rejection_names_a_loop_break_on_width_or_peak():
    tables = _finalist_like()
    tables["0.27 nA"][1]["cycle_ms"] = 3.
    tables["0.19 nA"][0]["width_ms"] = .4
    tables["0.27 nA"][5]["peak_mv"] = 30.
    reject = rejection(tables)
    assert reject["met"] and len(reject["loop_breaks"]) == 2
    assert "0.19 nA: cycle-1 width 0.4" in reject["loop_breaks"][0] and "0.27 nA: peak 30.0" in reject["loop_breaks"][1]


def _report(tables):
    rows = [{"input": "0.27 nA", "row": "rate_hz", "cycle": None, "verdict": "fail", "contract": "no contract row"},
            {"input": "0.27 nA", "row": "width_ms", "cycle": 1, "verdict": "pass", "contract": "pass"},
            {"input": "0.27 nA", "row": "ahp_mv", "cycle": 2, "verdict": "pass", "contract": "fail"},
            {"input": "0.19 nA", "row": "ahp_mv", "cycle": 1, "verdict": "unresolvable", "contract": "pass"}]
    return {"counts": {"0.19 nA": {"human": 12, "model": 12}, "0.27 nA": {"human": 43, "model": 37}},
            "rows": rows, "tables": {label: {"model": table} for label, table in tables.items()}}


def test_tier_verdicts_separate_usable_from_contract():
    tiers = tier_verdicts(_report(_finalist_like()))
    assert tiers["0.27 nA"]["usable_verdict"] == "fail" and tiers["0.27 nA"]["usable"]["fail"] == ["rate_hz"]
    assert tiers["0.27 nA"]["contract"] == {"count": "FAIL", "failing_rows": ["ahp_mv2"], "verdict": "FAIL"}
    assert tiers["0.19 nA"]["usable_verdict"] == "pass" and tiers["0.19 nA"]["usable"]["unresolvable"] == ["ahp_mv1"]
    assert tiers["0.19 nA"]["contract"]["verdict"] == "pass"


def test_decide_fails_at_cap_on_the_finalist_and_passes_on_a_held_run():
    initiation = {"019": {"axon_first": True, "spikes": 14}, "027": {"axon_first": True, "spikes": 37}}
    log = [{"name": "g-natg-axon15", "input": "019", "seconds": 40., "returncode": 0, "aborted": False, "command": []},
           {"name": "other", "input": "019", "seconds": 1., "returncode": 0, "aborted": False, "command": []}]
    failed = decide(_report(_finalist_like()), initiation, log, "abc")
    assert failed["verdict"] == "FAIL" and failed["decision"].startswith("FAIL at cap: rejection clause met")
    assert failed["bands_total"] == 14 and failed["runs"] == [{"input": "019", "seconds": 40., "returncode": 0, "aborted": False}]
    held = {"0.19 nA": _table([20., 3., 20.] + [60.]*13, [-80.1, -80.5, -80.7] + [-80.8]*13),
            "0.27 nA": _table([13., 3., 12.] + [22.]*38, [-79., -79.3, -79.6] + [-80.]*38)}
    passed = decide(_report(held), initiation, log, "abc")
    assert passed["verdict"] == "PASS" and passed["bands_held"] == 14
    lost = decide(_report(held), {"019": {"axon_first": False}, "027": {"axon_first": True}}, log, "abc")
    assert lost["verdict"] == "FAIL" and "axon-first" in lost["decision"]
