"""Tests for the C3 scan watchdog with a fake child; no network, no GCS."""

import json
import sys
import textwrap

from docs.evidence.h01_c3_scan_watchdog import checkpoint_cells, count_progress_lines, main

FAKE_CHILD = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path
    output = Path(sys.argv[sys.argv.index("--output")+1])
    ckpt = output.with_suffix(".cells.json")
    cells = json.loads(ckpt.read_text())["cells"] if ckpt.exists() else {}
    mode = os.environ["FAKE_MODE"]
    if mode == "fail":
        print("boom", flush=True)
        sys.exit(2)
    n = len(cells)
    cells[str(n)] = {"c3_ids": []}
    ckpt.write_text(json.dumps({"key": {"limit": 3000, "archive": "a.zip"}, "cells": cells}))
    print("cell", n+1, "1234", 5, "c3 ids", 0.1, "s", flush=True)
    if mode == "stall":
        time.sleep(60)
    cells[str(n+1)] = {"c3_ids": []}
    ckpt.write_text(json.dumps({"key": {"limit": 3000, "archive": "a.zip"}, "cells": cells}))
    print("cell", n+2, "1235", 5, "c3 ids", 0.1, "s", flush=True)
    print("pre_synaptic_cell 3 annotations", flush=True)
    print(json.dumps({"cells": n+2}), "wall", 0.3, "s", flush=True)
    output.write_text("{}")
''')


def test_progress_regex_accepts_child_lines_only():
    text = "cell 27 5654281423 66 c3 ids 98.4 s\npost_synaptic_cell 812 annotations\n" \
           '{"cells": 104} wall 950.5 s\nsome warning\ncell 28 in progress\n'
    assert count_progress_lines(text) == 3


def run(tmp_path, monkeypatch, mode, **overrides):
    monkeypatch.setenv("FAKE_MODE", mode)
    script = tmp_path/"fake_child.py"
    script.write_text(FAKE_CHILD)
    output = tmp_path/"edges-3000.json"
    argv = ["--python", sys.executable, "--script", str(script), "--output", str(output),
            "--record", str(tmp_path/"watchdog.json"), "--log-dir", str(tmp_path), "--root", str(tmp_path),
            "--cache", str(tmp_path), "--poll-seconds", "0.2", "--stall-seconds", str(overrides.get("stall", 1.5)),
            "--max-relaunches", str(overrides.get("relaunches", 3))]
    summary = main(argv)
    assert json.loads((tmp_path/"watchdog.json").read_text())["statement"] == summary["statement"]
    return summary


def test_finishing_child_is_one_launch_and_complete(tmp_path, monkeypatch):
    summary = run(tmp_path, monkeypatch, "finish")
    assert [x["outcome"] for x in summary["launches"]] == ["exit"]
    assert summary["launches"][0]["exit_code"] == 0 and summary["launches"][0]["progress_lines"] == 4
    assert summary["statement"] == "2 of 104 cells scanned" and summary["status"] == "complete"
    assert summary["launches"][0]["started"] <= summary["launches"][0]["ended"]


def test_stalling_child_is_killed_and_relaunched_up_to_the_cap(tmp_path, monkeypatch):
    summary = run(tmp_path, monkeypatch, "stall", relaunches=2)
    assert [x["outcome"] for x in summary["launches"]] == ["killed_stall"]*3
    assert summary["kills"] == 3 and all(x["seconds_without_progress"] >= 1.5 for x in summary["launches"])
    assert [(x["cells_before"], x["cells_after"]) for x in summary["launches"]] == [(0, 1), (1, 2), (2, 3)]
    assert summary["statement"] == "3 of 104 cells scanned" and summary["status"].startswith("partial")
    assert checkpoint_cells(tmp_path/"edges-3000.cells.json") == 3


def test_deterministic_failure_stops_after_two_exits_without_progress(tmp_path, monkeypatch):
    summary = run(tmp_path, monkeypatch, "fail")
    assert [x["exit_code"] for x in summary["launches"]] == [2, 2]
    assert "two consecutive exits without progress" in summary["status"]
    assert summary["statement"] == "0 of 104 cells scanned"
