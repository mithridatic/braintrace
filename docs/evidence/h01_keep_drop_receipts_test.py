"""Tests for the keep/drop receipt copier."""

import json
import sys

import numpy as np
import pytest

from docs.evidence import h01_keep_drop_receipts as receipts


def _run(folder, name, terminated=True, n=1000):
    src = folder/name
    src.mkdir()
    time_ms = np.arange(n)*.005
    events = np.zeros((n, 1), bool)
    events[[7, 500]] = True
    np.savez(src/"run.npz", time_ms=time_ms, voltage=-70.+time_ms, output_voltage=-65.+time_ms, events=events)
    (src/"launch.json").write_text("{}")
    (src/"run.json").write_text(json.dumps(dict(cell=name)))
    if terminated:
        (src/"terminal.json").write_text('{"exit_code": 0}')
    return src


def test_decimate_keeps_every_nth_sample_and_exact_spike_times(tmp_path):
    src = _run(tmp_path, "transfer-all-1")
    with np.load(src/"run.npz") as npz:
        out = receipts.decimate(npz, 100)
    assert out["time_ms"].shape == (10,) and out["voltage"][1] == pytest.approx(-70.+.5)
    assert out["spike_times_ms"].tolist() == pytest.approx([.035, 2.5]) and int(out["every"]) == 100
    with np.load(src/"run.npz") as npz, pytest.raises(ValueError):
        receipts.decimate(npz, 0)


def test_copy_all_skips_unterminated_runs_and_writes_index(tmp_path, monkeypatch):
    _run(tmp_path, "transfer-all-1")
    _run(tmp_path, "transfer-all-1-dthalf")
    _run(tmp_path, "transfer-all-2", terminated=False)
    out = tmp_path/"out"
    monkeypatch.setattr(sys, "argv", ["x", "--folder", str(tmp_path), "--out-dir", str(out), "--every", "50"])
    receipts.main()
    index = json.loads((out/"index.json").read_text())
    assert index == dict(every=50, runs=["transfer-all-1", "transfer-all-1-dthalf"])
    assert not (out/"transfer-all-2").exists()
    assert json.loads((out/"transfer-all-1"/"run.json").read_text()) == dict(cell="transfer-all-1")
    with np.load(out/"transfer-all-1-dthalf"/"trace.npz") as trace:
        assert trace["time_ms"].shape == (20,)
