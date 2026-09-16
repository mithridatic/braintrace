"""Tests for the kept-network driven-window registration."""

import json
import sys

from docs.evidence import h01_keep_drop_window as window

INPUTS = dict(topology="t.json", components="c.json", currents="cur.json")


def test_plan_carries_the_gate_fields_and_step_count():
    control = window.plan(window.RUNS[1], 7, INPUTS, "abc")
    assert control["cells"] == 7 and control["control"] == "ei" and control["steps"] == 80000
    assert control["pulse_window_ms"] == [2., 40.] and control["topology"] == "t.json"
    fine = window.plan(window.RUNS[6], 7, INPUTS, "abc")
    assert fine["dt_ms"] == .0003125 and fine["steps"] == 32000


def test_command_uses_per_cell_currents_and_the_registered_pulse():
    line = window.command(window.RUNS[4], 7, INPUTS)
    assert "--currents-json cur.json" in line and "--pulse-delay-ms 2 --pulse-duration-ms 38" in line
    assert "--control disconnected" in line and "--cells 7" in line and "--current-na" not in line
    assert line.startswith("$R kept-ctrl-disconnected-50ms 14400 --")


def test_queue_runs_bench_alone_then_pairs():
    script = window.queue_script(7, INPUTS)
    lines = script.splitlines()
    assert lines[6].startswith("CPUSET=0-63 $R kept-bench-ei-1ms")
    assert script.count('run 0-63 "') == 3 and script.count('run 64-127 "') == 3
    assert script.count("wait_done kept-") == 6 and lines[-1].startswith("echo QUEUE-DONE")


def test_cli_writes_seven_plans_and_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--cells", "5", "--source-commit", "abc", "--out-dir", str(tmp_path)])
    window.main()
    plans = sorted(tmp_path.glob("plan-*.json"))
    assert len(plans) == 7 and all(json.loads(p.read_text())["cells"] == 5 for p in plans)
    assert (tmp_path/"kept-window-queue.sh").read_text().count("\n") > 10
