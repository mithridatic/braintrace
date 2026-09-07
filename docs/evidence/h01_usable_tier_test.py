import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_usable_tier as tier  # noqa: E402


def synthetic(spike_times, stop_ms=400., rest=-70., ahp=-80.):
    """Trace with triangular spikes at the given times and a recovering minimum after each."""
    time = np.arange(0., stop_ms, .02)
    voltage = np.full_like(time, rest)
    for t in spike_times:
        rise = (time >= t) & (time < t+.4)
        fall = (time >= t+.4) & (time < t+1.2)
        recover = (time >= t+1.2) & (time < t+6.)
        voltage[rise] = rest+(20.-rest)*(time[rise]-t)/.4
        voltage[fall] = 20.-(20.-ahp)*(time[fall]-t-.4)/.8
        voltage[recover] = ahp+(rest-ahp)*(time[recover]-t-1.2)/4.8
    return time, voltage


def test_cycle_table_retains_one_row_per_cycle_with_width():
    table = tier.cycle_table(*synthetic([50., 60., 80., 110.]), (40., 200.), "x")
    assert [r["cycle"] for r in table] == [1, 2, 3, 4]
    assert all(r["input"] == "x" for r in table)
    assert table[1]["cycle_ms"] == pytest.approx(10., abs=.1)
    assert 0.4 < table[0]["width_ms"] < 0.7
    assert table[0]["ahp_mv"] == pytest.approx(-80., abs=.5)


def test_qc_view_rate_from_full_cycles_and_adaptation_ratio():
    view = tier.qc_view(tier.cycle_table(*synthetic([50., 60., 80., 110.]), (40., 200.), "x"), (40., 200.))
    assert view["count"] == 4
    assert view["rate_hz"] == pytest.approx(1000./20., rel=.02)
    assert view["adaptation_ratio"] == pytest.approx(3., rel=.02)
    assert view["count_rate_hz"] == pytest.approx(25.)


def test_qc_view_without_enough_cycles():
    view = tier.qc_view(tier.cycle_table(*synthetic([50.]), (40., 200.), "x"), (40., 200.))
    assert view["rate_hz"] == 0. and view["adaptation_ratio"] is None


def test_spreads_use_dixon_and_cyclical_factors():
    assert tier.repeat_spread([1., 2., None]) == pytest.approx(2.95)
    assert tier.repeat_spread([1., 2., 3., 4., 5.]) == pytest.approx(4*1.47)
    assert tier.repeat_spread([1.]) is None
    assert tier.cyclical_spread([1., 3.]) == pytest.approx(2*1.47)
    assert tier.cyclical_spread([1.]) is None


def test_usable_limits_fall_back_to_cyclical_spread():
    view = {"rate_hz": 40., "adaptation_ratio": 4., "width_ms": [1., 1.2], "ahp_mv": [-80., -79.]}
    limits = tier.usable_limits(view, [{"width_ms": 1., "ahp_mv": -80.}, {"width_ms": 1.1, "ahp_mv": -81.}])
    assert limits["rate_hz"]["limit"] == pytest.approx(6.)
    assert limits["adaptation_ratio"]["limit"] == pytest.approx(1.)
    assert limits["width_ms"]["basis"].endswith("spread from human repeats")
    assert limits["width_ms"]["limit"] == pytest.approx(.2*1.1)
    assert limits["width_ms"]["spread"] == pytest.approx(.1*2.95)
    assert limits["ahp_mv"]["limit"] == 2.
    fallback = tier.usable_limits(view, [{}])
    assert fallback["ahp_mv"]["basis"].endswith("spread from the human cyclical range")
    assert fallback["ahp_mv"]["spread"] == pytest.approx(1.47)
    assert tier.usable_limits({**view, "adaptation_ratio": None}, [])["adaptation_ratio"]["limit"] is None


def test_verdict_states():
    assert tier.verdict(1., 1.5, 1., None) == "pass"
    assert tier.verdict(1., 2.5, 1., None) == "fail"
    assert tier.verdict(1., 1.1, 1., .6) == "unresolvable"
    assert tier.verdict(None, 1., 1., None) == "unavailable"
    assert tier.verdict(1., 1., None, None) == "unavailable"


def test_contract_verdict_only_for_contract_rows():
    assert tier.contract_verdict("ahp_mv", -80., -79.5) == "pass"
    assert tier.contract_verdict("width_ms", 1., 1.2) == "fail"
    assert tier.contract_verdict("rate_hz", 1., 1.) == "no contract row"
    assert tier.contract_verdict("ahp_mv", None, 1.) == "no contract row"


def test_compare_matches_cycles_and_reports_both_tiers():
    human = {"rate_hz": 40., "adaptation_ratio": 4., "width_ms": [1., 1.1, 1.2], "ahp_mv": [-80., -79.]}
    model = {"rate_hz": 30., "adaptation_ratio": 4.5, "width_ms": [1.02, 1.3], "ahp_mv": [-80.5, -78.]}
    limits = {"rate_hz": {"limit": 6., "spread": None, "basis": "b"},
              "adaptation_ratio": {"limit": 1., "spread": None, "basis": "b"},
              "width_ms": {"limit": .1, "spread": .02, "basis": "b"},
              "ahp_mv": {"limit": 1., "spread": .8, "basis": "b"}}
    rows = tier.compare("in", human, model, limits)
    assert [r["row"] for r in rows] == ["rate_hz", "adaptation_ratio", "width_ms", "width_ms", "ahp_mv", "ahp_mv"]
    assert [r["verdict"] for r in rows] == ["fail", "pass", "pass", "fail", "unresolvable", "unresolvable"]
    assert rows[2]["contract"] == "pass" and rows[3]["contract"] == "fail" and rows[0]["contract"] == "no contract row"


def test_render_lists_counts_verdicts_and_tables():
    report = {"cell": "I", "model": "m", "counts": {"a": {"human": 3, "model": 2}},
              "rows": [{"input": "a", "row": "rate_hz", "cycle": None, "human": 40., "model": 30., "limit": 6.,
                        "spread": None, "verdict": "fail", "contract": "no contract row", "basis": "b"}],
              "tables": {"a": {"human": tier.cycle_table(*synthetic([50., 60.]), (40., 200.), "a")}}}
    text = tier.render(report)
    assert "| a | 3 | 2 | FAIL |" in text
    assert "| a | rate_hz |  | 40 | 30 | 6 |  | fail | no contract row | b |" in text
    assert "### a, human" in text and text.count("| 1 |") >= 1


def test_main_writes_both_files(monkeypatch, tmp_path):
    monkeypatch.setattr(tier, "EVIDENCE", tmp_path)
    monkeypatch.setattr(tier, "score_cell", lambda cell, spec: {"cell": cell, "model": "m", "counts": {},
                                                                 "repeats": [], "rows": [], "tables": {}})
    tier.main(["--cell", "I"])
    assert (tmp_path/"h01-usable-tier-i.json").exists() and (tmp_path/"h01-usable-tier-i.md").exists()
