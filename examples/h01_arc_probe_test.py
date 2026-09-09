"""Failure evidence and resource-limit validation for the diagnostic runner."""

import json
import sys

import pytest

from examples import h01_arc_probe


def test_failed_construction_keeps_phase_evidence(tmp_path, monkeypatch):
    output = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--cells", "4", "--cache", str(tmp_path),
                                     "--output", str(output)])
    monkeypatch.setattr(h01_arc_probe, "H01Archive", lambda _: object())
    monkeypatch.setattr(h01_arc_probe, "H01Annotations", lambda _: object())
    def fail(*args, **kwargs):
        raise ValueError("invalid pinned population")
    monkeypatch.setattr(h01_arc_probe, "make_h01_network", fail)
    h01_arc_probe.main()
    result = json.loads(output.read_text())
    assert result["status"] == "blocked"
    assert result["error"] == "invalid pinned population"
    assert result["phases"]["construction"]["status"] == "failed"
    assert result["phases"]["construction"]["seconds"] >= 0
    assert result["limits"] == {"wall_seconds": 900., "rss_gib": 16.}


@pytest.mark.parametrize("flag,value", [("--rss-limit-gib", "0"),
    ("--rss-limit-gib", "nan"), ("--wall-limit-seconds", "-1")])
def test_invalid_limits_rejected_before_construction(tmp_path, monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["probe", "--cells", "4", "--cache", str(tmp_path),
        "--output", str(tmp_path/"unused.json"), flag, value])
    with pytest.raises(SystemExit) as exc:
        h01_arc_probe.main()
    assert exc.value.code == 2
    assert not (tmp_path/"unused.json").exists()
