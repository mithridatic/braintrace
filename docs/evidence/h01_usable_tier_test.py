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
    assert limits["width_ms"]["limit"] is None
    assert limits["width_ms"]["fraction"] == .2
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


@pytest.mark.parametrize("spread,expected", [(0., ["fail", "pass"]),
                                            (.11, ["unresolvable", "pass"])])
def test_width_allowance_and_resolution_use_each_human_cycle(spread, expected):
    human = {"rate_hz": 10., "adaptation_ratio": 2.,
             "width_ms": [1., 2.], "ahp_mv": [-80., -80.]}
    model = {**human, "width_ms": [1.25, 1.7]}
    limits = tier.usable_limits(human, [])
    limits["width_ms"]["spread"] = spread
    rows = [r for r in tier.compare("x", human, model, limits) if r["row"] == "width_ms"]
    assert [r["limit"] for r in rows] == pytest.approx([.2, .4])
    assert [r["verdict"] for r in rows] == expected
    assert [r["contract"] for r in rows] == ["fail", "fail"]


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
    monkeypatch.setattr(tier, "score_cell", lambda cell, spec, root=None: {"cell": cell, "model": "m", "counts": {},
                                                                 "repeats": [], "rows": [], "tables": {}})
    tier.main(["--cell", "I"])
    assert (tmp_path/"h01-usable-tier-i.json").exists() and (tmp_path/"h01-usable-tier-i.md").exists()


def test_score_cell_keeps_input_counts_and_reads_real_repeat_files(tmp_path, monkeypatch):
    """The file-to-report path preserves mismatched counts and per-cycle limits."""
    import h5py

    monkeypatch.setattr(tier, "EVIDENCE", tmp_path)
    time, human = synthetic([50., 70., 100.])
    _, extra = synthetic([50., 70., 100., 140.])
    for label, voltage in (("a", human), ("b", extra)):
        np.savez(tmp_path/f"{label}-human.npz", time_ms=time, corrected_voltage_mv=human)
        np.savez(tmp_path/f"{label}-model.npz", time_ms=time, voltage_mv=voltage)
    with h5py.File(tmp_path/"repeats.nwb", "w") as nwb:
        for sweep, voltage in ((1, human), (2, human), (3, np.full_like(human, -70.))):
            group = nwb.create_group(f"acquisition/timeseries/Sweep_{sweep}")
            group.create_dataset("data", data=(voltage+14.)/1000.)
            group.create_dataset("starting_time", data=0.).attrs["rate"] = 50000.
    spec = {"model": "fixture", "pulse_ms": (40., 200.),
            "repeats": ("repeats.nwb", (1, 2, 3), (40., 200.)),
            "inputs": {label: (f"{label}-human.npz", f"{label}-model") for label in ("a", "b")}}
    report = tier.score_cell("E", spec, root=tmp_path)
    assert report["counts"] == {"a": {"human": 3, "model": 3},
                                "b": {"human": 3, "model": 4}}
    assert report["repeats"][-1] == {}
    assert len(report["tables"]["b"]["model"]) == 4
    widths = [r for r in report["rows"] if r["row"] == "width_ms"]
    assert len(widths) == 6
    assert all(r["limit"] == pytest.approx(.2*r["human"]) for r in widths)
    assert "| b | 3 | 4 | FAIL |" in tier.render(report)


def test_rate_limit_is_unavailable_without_human_spikes():
    """A subthreshold input (sweep 43) must not pass the rate row on 0 == 0."""
    view = {"rate_hz": 0., "adaptation_ratio": None, "width_ms": [], "ahp_mv": []}
    limits = tier.usable_limits(view, [])
    assert limits["rate_hz"]["limit"] is None
    rows = tier.compare("110 pA", view, view, limits)
    assert [r["verdict"] for r in rows] == ["unavailable", "unavailable"]


def test_qc_view_reports_first_spike_latency_from_the_pulse_start():
    table = tier.cycle_table(*synthetic([50., 60.]), (40., 200.), "x")
    assert "peak_ms" in table[0]
    assert tier.qc_view(table, (40., 200.))["first_spike_ms"] == pytest.approx(10.4, abs=.1)
    assert tier.qc_view([], (40., 200.))["first_spike_ms"] is None


def test_repeat_rows_use_the_repeat_range_and_make_a_zero_range_exact():
    repeats = [{"repeat_count": 1, "first_spike_ms": 10.}, {"repeat_count": 1, "first_spike_ms": 12.},
               {"repeat_count": 1, "first_spike_ms": 11.}, {"repeat_count": 1, "first_spike_ms": 10.5},
               {"repeat_count": 1, "first_spike_ms": 11.5}, {}]
    human = {"count": 1, "first_spike_ms": 10.}
    rows = tier.repeat_rows("200 pA", human, {"count": 1, "first_spike_ms": 12.5}, repeats)
    assert [r["row"] for r in rows] == ["count", "first_spike_ms"]
    assert rows[0]["limit"] == 0. and rows[0]["verdict"] == "pass" and rows[0]["contract"] == "pass"
    assert rows[1]["limit"] == pytest.approx(2.*1.47)
    assert rows[1]["verdict"] == "pass" and rows[1]["contract"] == "no contract row"
    assert rows[1]["basis"].startswith("human repeat range x DLF over 6 repeats")
    extra = tier.repeat_rows("200 pA", human, {"count": 2, "first_spike_ms": 14.}, repeats)
    assert [r["verdict"] for r in extra] == ["fail", "fail"]
    assert extra[0]["contract"] == "fail"
    missing = tier.repeat_rows("200 pA", human, {"count": 0, "first_spike_ms": None}, repeats)
    assert [r["verdict"] for r in missing] == ["fail", "unavailable"]
    assert tier.repeat_rows("x", human, human, [{}])[0]["verdict"] == "unavailable"


def test_parse_input_and_resolve_spec_fill_candidate_and_override_inputs():
    assert tier.parse_input("200 pA=.cache/h.npz@h01-e-gain/g0-b3-sweep56") == (
        "200 pA", (".cache/h.npz", "h01-e-gain/g0-b3-sweep56"))
    for bad in ("200 pA", "200 pA=only-human", "=a@b", "x=@b", "x=a@"):
        with pytest.raises(ValueError, match="LABEL=HUMAN_NPZ@MODEL_STEM"):
            tier.parse_input(bad)
    spec = tier.resolve_spec(tier.CELLS["E-gain"], "g1-ih-half")
    assert spec["inputs"]["200 pA"] == (".cache/human-pyramidal-l2/sweep-56.npz", "h01-e-gain/g1-ih-half-sweep56")
    assert spec["output"] == "h01-e-gain/g1-ih-half-usable" and "g1-ih-half" in spec["model"]
    assert spec["repeat_inputs"] == {"200 pA": (56, 59, 60, 61, 62)}
    assert set(spec["inputs"]) == {"110 pA", "200 pA", "250 pA", "310 pA"}
    overridden = tier.resolve_spec(tier.CELLS["E"], inputs=["a=h.npz@m", "b=h2.npz@m2"])
    assert overridden["inputs"] == {"a": ("h.npz", "m"), "b": ("h2.npz", "m2")}
    assert "output" not in overridden and tier.CELLS["E"]["inputs"]["250 pA"][1].startswith("h01-e-energetic")


def test_score_cell_adds_repeat_rows_for_a_repeat_input(tmp_path, monkeypatch):
    import h5py

    monkeypatch.setattr(tier, "EVIDENCE", tmp_path)
    time, one = synthetic([50.])
    _, two = synthetic([50., 90.])
    _, flat = synthetic([])
    np.savez(tmp_path/"h.npz", time_ms=time, corrected_voltage_mv=one)
    np.savez(tmp_path/"m.npz", time_ms=time, voltage_mv=two)
    np.savez(tmp_path/"h0.npz", time_ms=time, corrected_voltage_mv=flat)
    np.savez(tmp_path/"m0.npz", time_ms=time, voltage_mv=flat)
    with h5py.File(tmp_path/"repeats.nwb", "w") as nwb:
        for sweep in (1, 2, 3):
            group = nwb.create_group(f"acquisition/timeseries/Sweep_{sweep}")
            group.create_dataset("data", data=(one+14.)/1000.)
            group.create_dataset("starting_time", data=0.).attrs["rate"] = 50000.
    spec = {"model": "fixture", "pulse_ms": (40., 200.), "repeats": ("repeats.nwb", (1, 2, 3), (40., 200.)),
            "repeat_inputs": {"200 pA": (1, 2, 3)},
            "inputs": {"200 pA": ("h.npz", "m"), "110 pA": ("h0.npz", "m0")}}
    report = tier.score_cell("E-gain", spec, root=tmp_path)
    assert report["repeats"][0]["repeat_count"] == 1
    by_row = {(r["input"], r["row"]): r for r in report["rows"]}
    assert by_row[("200 pA", "count")]["verdict"] == "fail" and by_row[("200 pA", "count")]["limit"] == 0.
    assert by_row[("200 pA", "first_spike_ms")]["verdict"] == "pass"
    assert by_row[("200 pA", "rate_hz")]["verdict"] == "unavailable"
    assert by_row[("110 pA", "rate_hz")]["verdict"] == "unavailable"
    assert ("110 pA", "count") not in by_row
    assert report["counts"] == {"200 pA": {"human": 1, "model": 2}, "110 pA": {"human": 0, "model": 0}}
    text = tier.render(report)
    assert "| 200 pA | count |  | 1 | 2 | 0 |  | fail | fail |" in text


def test_main_accepts_candidate_root_inputs_and_output(monkeypatch, tmp_path):
    seen = {}

    def fake(cell, spec, root):
        seen.update(cell=cell, spec=spec, root=root)
        return {"cell": cell, "model": spec["model"], "counts": {}, "repeats": [], "rows": [], "tables": {}}
    monkeypatch.setattr(tier, "EVIDENCE", tmp_path)
    monkeypatch.setattr(tier, "score_cell", fake)
    tier.main(["--cell", "E-gain", "--candidate", "g0-b3", "--root", str(tmp_path)])
    assert (tmp_path/"h01-e-gain/g0-b3-usable.json").exists() and (tmp_path/"h01-e-gain/g0-b3-usable.md").exists()
    assert seen["root"] == tmp_path and seen["spec"]["inputs"]["250 pA"][1] == "h01-e-gain/g0-b3-sweep50"
    tier.main(["--cell", "E", "--inputs", "350 pA=.cache/x.npz@h01-e-usable/b3-sk035-ca-decay-sweep55",
               "--output", "custom/stem"])
    assert seen["spec"]["inputs"] == {"350 pA": (".cache/x.npz", "h01-e-usable/b3-sk035-ca-decay-sweep55")}
    assert (tmp_path/"custom/stem.md").exists()
    with pytest.raises(SystemExit) as error:
        tier.main(["--cell", "E", "--inputs", "broken"])
    assert error.value.code == 2
