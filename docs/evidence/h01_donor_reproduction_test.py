import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_donor_reproduction as repro  # noqa: E402


def trace(spike_times, stop_ms=400.):
    time = np.arange(0., stop_ms, .02)
    voltage = np.full_like(time, -70.)
    for t in spike_times:
        voltage[(time >= t) & (time < t+1.)] = 20.
    return time, voltage


def test_crossings_count_inside_the_pulse_and_time_the_first_from_onset():
    result = repro.crossings(*trace([30., 50., 60., 250.]), (40., 200.))
    assert result["count"] == 2
    assert result["first_spike_ms"] == pytest.approx(10., abs=.03)
    assert result["finite"] and result["end_ms"] == pytest.approx(399.98)
    assert repro.crossings(*trace([]), (40., 200.))["first_spike_ms"] is None


def test_count_verdict_states():
    assert repro.count_verdict(14, 14, (14, 14, 13, 12)) == {"exact": "held", "repeat_band": "held"}
    assert repro.count_verdict(13, 14, (14, 14, 13, 12)) == {"exact": "missed_within_one", "repeat_band": "held"}
    assert repro.count_verdict(16, 14, (14, 14, 13, 12)) == {"exact": "rejected", "repeat_band": "missed"}
    assert repro.count_verdict(35, 34, None) == {"exact": "missed_within_one", "repeat_band": None}
    assert repro.count_verdict(None, 34, None) == {"exact": "untested", "repeat_band": "untested"}


def test_pair_limit_and_latency_verdict():
    assert repro.pair_limit(10., 10.2) == pytest.approx(.2*2.95)
    assert repro.pair_limit(None, 1.) is None
    assert repro.latency_verdict(24.3, 24.16, .59) == "held"
    assert repro.latency_verdict(26.9, 24.16, .59) == "missed"
    assert repro.latency_verdict(None, 24.16, .59) == "untested"
    assert repro.latency_verdict(24.3, 24.16, None) == "untested"


def _write(folder, name, spikes, stop):
    time, voltage = trace(spikes, stop)
    np.savez(folder/f"{name}.npz", time_ms=time, voltage_mv=voltage)


def test_score_donor_builds_rows_wall_clocks_and_overall_verdict(tmp_path, monkeypatch):
    evidence, root = tmp_path/"evidence", tmp_path/"root"
    folder = evidence/"h01-x-reproduction"
    folder.mkdir(parents=True)
    (root/".cache").mkdir(parents=True)
    time, voltage = trace([60., 100., 150.], 400.)
    np.savez(root/".cache/human.npz", time_ms=time, corrected_voltage_mv=voltage)
    _write(folder, "source-1", [60.1, 100., 150.], 400.)
    _write(folder, "source-dt-half-1", [60.2, 100., 150.], 400.)
    _write(folder, "source-2", [61., 100., 150., 180., 190.], 400.)
    _write(folder, "source-dt-half-2", [61., 100., 150., 180.], 400.)
    (folder/"campaign-log.json").write_text(json.dumps([
        {"name": "source", "input": "1", "seconds": 7.5, "returncode": 0, "aborted": False},
        {"name": "source", "input": "2", "seconds": 14., "returncode": 0, "aborted": False}]))
    (folder/"source-1.stderr.txt").write_text("", encoding="utf-8")
    monkeypatch.setitem(repro.DONORS, "x", {
        "donor_key": "k", "output_dir": "h01-x-reproduction", "manifest": "m", "decision": "d", "specification": "s",
        "inputs": {"a": {"input": "1", "human": ".cache/human.npz", "sweep": 1, "pulse_ms": (40., 200.),
                         "registered_count": 3, "repeat_counts": (3, 3, 2), "registered_first_spike_ms": 20.},
                   "b": {"input": "2", "human": ".cache/human.npz", "sweep": 2, "pulse_ms": (40., 200.),
                         "registered_count": 3, "repeat_counts": None, "registered_first_spike_ms": 20.}}})
    report = repro.score_donor("x", root=root, evidence=evidence)
    a, b = report["inputs"]
    assert a["human_reread"]["count"] == 3 and a["source"]["count"] == 3 and a["source_dt_half"]["count"] == 3
    assert a["first_spike_pair_limit_ms"] == pytest.approx(.1*2.95, abs=.01)
    assert a["verdicts"] == {"count": {"exact": "held", "repeat_band": "held"}, "dt_pair_count": "agrees",
                             "first_spike": "held"}
    assert b["verdicts"]["dt_pair_count"] == "time level open" and b["verdicts"]["first_spike"] == "time level open"
    assert b["verdicts"]["count"]["exact"] == "rejected"
    assert report["reproduction_verdict"] == "time_level_open"
    assert report["wall_clocks"]["total_seconds"] == pytest.approx(21.5)
    assert report["wall_clocks"]["stderr_files"] == {"source-1.stderr.txt": ""}
    text = repro.render(report)
    assert "**Reproduction verdict: time_level_open**" in text and "| source | 1 | 7.5 | 0 | False |" in text


def test_overall_orders_untested_rejected_and_missed():
    def inputs(*states):
        return [{"verdicts": s} for s in states]
    held = {"count": {"exact": "held"}, "dt_pair_count": "agrees", "first_spike": "held"}
    assert repro.overall(inputs(held, held)) == "reproduced"
    assert repro.overall(inputs(held, {**held, "first_spike": "missed"})) == "prediction_missed"
    assert repro.overall(inputs(held, {**held, "count": {"exact": "missed_within_one"}})) == "prediction_missed"
    assert repro.overall(inputs(held, {**held, "count": {"exact": "rejected"}})) == "rejected"
    assert repro.overall(inputs(held, {**held, "count": {"exact": "untested"}})) == "untested"
    assert repro.wall_clocks(Path("does-not-exist"))["total_seconds"] is None


def test_main_writes_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "EVIDENCE", tmp_path)
    monkeypatch.setattr(repro, "score_donor", lambda name, root=None: {
        "donor_key": "k", "specification": "s", "manifest": "m", "decision_json": "d", "detector": "x", "pair_factor": 2.95,
        "inputs": [], "wall_clocks": {"runs": [], "total_seconds": None, "stderr_files": {}}, "reproduction_verdict": "untested"})
    repro.main(["--donor", "sst"])
    assert (tmp_path/"h01-sst-reproduction.json").exists() and (tmp_path/"h01-sst-reproduction.md").exists()
