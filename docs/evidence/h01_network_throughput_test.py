"""Parsing, step count, fit, decision limit, cap, watchdog, RSS sampling and reporting of the throughput benchmark."""

import json
import sys
from types import SimpleNamespace

import pytest

from docs.evidence import h01_network_throughput as bench


def _build_record(duration_ms, init_run, construction=163.3, dt_ms=.005):
    return {"construction_seconds": construction, "initialization_and_run_seconds": init_run,
            "dt_ms": dt_ms, "duration_ms": duration_ms, "execution": "finite compiled smoke run"}


def _run(label, duration_ms, init_run, solver="h01_staggered_scan", status="completed", rss=214.):
    return {"label": label, "solver": solver, "status": status, "duration_ms": duration_ms, "dt_ms": .005,
            "steps": bench.step_count(duration_ms, .005), "construction_seconds": 163.3,
            "initialization_and_run_seconds": init_run, "peak_rss_mb": rss, "wall_seconds": None if init_run is None else 163.3+init_run}


def test_step_count_rounds_to_nearest_step():
    assert bench.step_count(.05, .005) == 10
    assert bench.step_count(1., .005) == 200
    assert bench.step_count(10., .005) == 2000
    assert bench.step_count(.005, .005) == 1
    with pytest.raises(ValueError, match="dt"):
        bench.step_count(1., 0.)


def test_parse_build_record(tmp_path):
    path = tmp_path/"network-build.json"
    path.write_text(json.dumps(_build_record(1., 190.)))
    parsed = bench.parse_build_record(path)
    assert parsed == {"construction_seconds": 163.3, "initialization_and_run_seconds": 190., "dt_ms": .005,
                      "duration_ms": 1., "steps": 200}


def test_parse_build_record_without_run_is_incomplete(tmp_path):
    path = tmp_path/"network-build.json"
    path.write_text(json.dumps({"construction_seconds": 163.3, "execution": "constructed; not simulated"}))
    parsed = bench.parse_build_record(path)
    assert parsed["construction_seconds"] == 163.3
    assert parsed["initialization_and_run_seconds"] is None
    assert parsed["steps"] is None
    assert bench.parse_build_record(tmp_path/"missing.json") == bench.EMPTY_RECORD


def test_fit_linear_recovers_exact_line():
    steps = [10, 200, 200, 2000]
    times = [150.+.1*s for s in steps]
    fit = bench.fit_linear(steps, times)
    assert fit["a"] == pytest.approx(150.)
    assert fit["b"] == pytest.approx(.1)
    assert fit["residuals"] == pytest.approx([0., 0., 0., 0.])
    assert bench.predict(fit, 2000) == pytest.approx(350.)
    with pytest.raises(ValueError, match="two distinct"):
        bench.fit_linear([200, 200], [1., 2.])


def test_fit_linear_least_squares_residuals():
    fit = bench.fit_linear([0, 1, 2], [0., 1., 4.])
    assert fit["a"] == pytest.approx(-1/3)
    assert fit["b"] == pytest.approx(2.)
    assert max(abs(r) for r in fit["residuals"]) == pytest.approx(2/3)


def test_decision_limit_is_range_times_dlf():
    assert bench.decision_limit([190., 192.]) == pytest.approx(2.*2.95)
    assert bench.decision_limit([190.]) is None
    assert bench.decision_limit([]) is None
    assert bench.DECISION_LIMIT_FACTOR == 2.95


def test_repeat_slopes_use_anchor_point():
    runs = [_run("d0p05", .05, 157.), _run("d1", 1., 176.), _run("d1r", 1., 178.)]
    slopes = bench.repeat_slopes(runs)
    assert slopes == pytest.approx([19./190., 21./190.])
    assert bench.repeat_slopes(runs[:2]) == pytest.approx([19./190.])
    assert bench.repeat_slopes(runs[:1]) == []


def test_check_cap_counts_launched_runs():
    assert bench.check_cap([_run("a", .05, 1.)]*4) == 5
    with pytest.raises(RuntimeError, match="cap"):
        bench.check_cap([_run("a", .05, 1.)]*5)


def test_watchdog_rule():
    assert not bench.is_silent(last_progress=100., now=699., silence_seconds=600.)
    assert bench.is_silent(last_progress=100., now=700., silence_seconds=600.)


class _FakeProcess:
    def __init__(self, rss, children=()):
        self._rss, self._children = rss, list(children)

    def memory_info(self):
        return type("Info", (), {"rss": self._rss})()

    def children(self, recursive):
        assert recursive
        return self._children


def test_tree_rss_sums_children_and_tolerates_vanished():
    tree = _FakeProcess(100*2**20, [_FakeProcess(50*2**20), _FakeProcess(0)])
    assert bench.tree_rss_mb(tree) == pytest.approx(150.)

    class Gone(_FakeProcess):
        def memory_info(self):
            raise bench.psutil.NoSuchProcess(1)

    assert bench.tree_rss_mb(_FakeProcess(2**20, [Gone(0)])) == pytest.approx(1.)


def test_build_command_names_the_example_and_flags(tmp_path):
    command = bench.build_command("py.exe", tmp_path, .05, "h01_staggered_scan", tmp_path/"cache", tmp_path/"top.json", .005)
    assert command[:3] == ["py.exe", "-u", "-m"]
    assert "examples.h01_verified_network" in command
    assert command[command.index("--duration-ms")+1] == "0.05"
    assert command[command.index("--dt-ms")+1] == "0.005"
    assert command[command.index("--solver")+1] == "h01_staggered_scan"
    assert command[command.index("--output")+1] == str(tmp_path/"network")
    assert "--build" in command


def test_summarize_linear_case_accepts_prediction():
    runs = [_run("d0p05", .05, 158.), _run("d1", 1., 177.), _run("d1r", 1., 178.), _run("d10", 10., 356.)]
    summary = bench.summarize(runs, anchor_seconds=157., cap_seconds=900.)
    fit = summary["fit"]
    assert fit["points"] == 4
    assert summary["seconds_per_simulated_ms"] == pytest.approx(200.*fit["b"])
    assert summary["decision_limit_b"] == pytest.approx(bench.decision_limit(bench.repeat_slopes(runs)))
    assert summary["decision_limit_seconds"] == pytest.approx(2.95)
    assert summary["linear_within_limit"] is True
    assert summary["prediction_0p05_within_anchor"] is True
    assert summary["predicted_10ms_seconds"] == pytest.approx(bench.predict(fit, 2000))
    assert summary["run_10ms_allowed"] is True
    assert summary["peak_rss_mb"] == 214.


def test_summarize_rejects_nonlinear_and_flags_over_cap():
    runs = [_run("d0p05", .05, 158.), _run("d1", 1., 177.), _run("d1r", 1., 178.), _run("d10", 10., 700.)]
    summary = bench.summarize(runs, anchor_seconds=157., cap_seconds=300.)
    assert summary["linear_within_limit"] is False
    assert summary["run_10ms_allowed"] is False
    assert summary["max_abs_residual_seconds"] > summary["decision_limit_seconds"]


def test_summarize_excludes_killed_and_control_runs_from_fit():
    runs = [_run("d0p05", .05, 158.), _run("d1", 1., 177.), _run("d1r", 1., 178.),
            _run("d10", 10., None, status="killed"), _run("ctl", 1., 400., solver="staggered")]
    summary = bench.summarize(runs, anchor_seconds=157., cap_seconds=900.)
    assert summary["fit"]["points"] == 3
    assert summary["linear_within_limit"] is None
    assert summary["untested"] == ["d10"]
    assert summary["controls"][0]["label"] == "ctl"


def test_summarize_with_one_point_reports_no_fit():
    summary = bench.summarize([_run("d0p05", .05, 158.)], anchor_seconds=157., cap_seconds=900.)
    assert summary["fit"] is None
    assert summary["prediction_0p05_within_anchor"] is None
    assert summary["run_10ms_allowed"] is None


def test_render_markdown_lists_every_run(tmp_path):
    runs = [_run("d0p05", .05, 158.), _run("d1", 1., 177.), _run("d1r", 1., 178.), _run("d10", 10., None, status="killed")]
    summary = bench.summarize(runs, anchor_seconds=157., cap_seconds=900.)
    text = bench.render_markdown(runs, summary)
    for label in ("d0p05", "d1r", "d10", "killed"):
        assert label in text
    assert "seconds per simulated ms" in text
    assert "not tested" in text


def test_report_writes_json_and_markdown(tmp_path):
    for label, duration, init_run in (("d0p05", .05, 158.), ("d1", 1., 177.), ("d1r", 1., 178.)):
        directory = tmp_path/f"bench-{label}"
        directory.mkdir()
        (directory/"run.json").write_text(json.dumps(_run(label, duration, init_run)))
    written = bench.report(tmp_path, tmp_path/"out.json", anchor_seconds=157., cap_seconds=900.)
    assert written["runs"][0]["label"] == "d0p05"
    assert (tmp_path/"out.json").exists() and (tmp_path/"out.md").exists()
    assert json.loads((tmp_path/"out.json").read_text())["summary"]["fit"]["points"] == 3


def test_load_runs_sorts_by_start(tmp_path):
    for label, start in (("late", "2026-09-07T12:00:00"), ("early", "2026-09-07T11:00:00")):
        directory = tmp_path/f"bench-{label}"
        directory.mkdir()
        record = _run(label, .05, 1.)
        record["started_at"] = start
        (directory/"run.json").write_text(json.dumps(record))
    assert [r["label"] for r in bench.load_runs(tmp_path)] == ["early", "late"]


def _fake_child(tmp_path, body):
    script = tmp_path/"child.py"
    script.write_text(body)
    return [sys.executable, "-u", str(script)]


def _args(tmp_path, label, silence=600.):
    return SimpleNamespace(root=tmp_path, label=label, python=sys.executable, duration_ms=1., solver="h01_staggered_scan",
                           cache=tmp_path, topology=tmp_path/"top.json", dt_ms=.005, worktree=tmp_path, silence_seconds=silence)


CHILD_OK = """import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
print('progress', flush=True)
out.write_text(json.dumps({'construction_seconds': 2., 'initialization_and_run_seconds': 3., 'dt_ms': .005, 'duration_ms': 1.}))
"""


def test_run_once_with_stand_in_child_records_timings(tmp_path, monkeypatch):
    command = _fake_child(tmp_path, CHILD_OK)+[str(tmp_path/"bench-one"/"network-build.json")]
    monkeypatch.setattr(bench, "build_command", lambda *a: command)
    record = bench.run_once(_args(tmp_path, "one"))
    assert record["status"] == "completed"
    assert record["initialization_and_run_seconds"] == 3. and record["steps"] == 200
    assert record["peak_rss_mb"] > 0 and record["wall_seconds"] > 0 and record["exit_code"] == 0
    assert json.loads((tmp_path/"bench-one"/"run.json").read_text())["label"] == "one"
    assert "progress" in (tmp_path/"bench-one"/"child.log").read_text()


def test_run_once_marks_failure_and_kills_silent_child(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "build_command", lambda *a: _fake_child(tmp_path, "raise SystemExit(3)\n"))
    assert bench.run_once(_args(tmp_path, "bad"))["status"] == "failed"
    monkeypatch.setattr(bench, "build_command", lambda *a: _fake_child(tmp_path, "import time\ntime.sleep(60)\n"))
    killed = bench.run_once(_args(tmp_path, "slow", silence=0.))
    assert killed["status"] == "killed" and killed["wall_seconds"] < 30
    assert killed["duration_ms"] == 1. and killed["steps"] == 200 and killed["initialization_and_run_seconds"] is None


def test_run_once_respects_cap(tmp_path):
    for i in range(5):
        (tmp_path/f"bench-{i}").mkdir()
        (tmp_path/f"bench-{i}"/"run.json").write_text(json.dumps(_run(str(i), .05, 1.)))
    with pytest.raises(RuntimeError, match="cap"):
        bench.run_once(_args(tmp_path, "six"))


def test_main_report_subcommand(tmp_path):
    bench.main(["report", "--root", str(tmp_path), "--output", str(tmp_path/"r.json")])
    assert json.loads((tmp_path/"r.json").read_text())["summary"]["fit"] is None
