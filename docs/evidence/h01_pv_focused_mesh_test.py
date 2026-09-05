"""Exercise numerical acceptance decisions with direct synthetic voltage traces."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from docs.evidence import h01_pv_focused_mesh as audit


@pytest.mark.parametrize("factor", [None, 243, 729, 2187])
@pytest.mark.parametrize("case,expected", [
    ("same", "supported"),
    ("shift", "rejected"),
    ("count", "rejected"),
    ("late_shift", "rejected"),
    ("missing_late", "invalid_missing_events"),
    ("classification", "rejected"),
])
def test_direct_events_control_acceptance(tmp_path, monkeypatch, factor, case, expected):
    """Reject discrete timing failures even when the remaining events agree."""
    source = Path(__file__).with_name("h01-pv-regional-mesh-axon81.json")
    template = json.loads(source.read_text())
    names = (["h01-pv-calcium-response-004-300-027-space81", "h01-pv-regional-mesh-axon81"]
             if factor is None else [f"h01-pv-regional-mesh-axon{f}" for f in (factor // 3, factor)])
    for index, stem in enumerate(names):
        meta = copy.deepcopy(template)
        if factor is None:
            if index == 0:
                meta["refine_region"] = "all"
                for section in meta["geometry"]:
                    if ".axon[" not in section["name"]:
                        section["nseg"] *= 9
        else:
            mesh = factor // 3 if index == 0 else factor
            meta["nseg_factor"] = mesh
            for section in meta["geometry"]:
                if ".axon[" in section["name"]:
                    section["nseg"] *= mesh // 81
        (tmp_path / (stem + ".json")).write_text(json.dumps(meta))
        starts = [280., 300., 320., 340., 1010., 1040.]
        if index == 1:
            if case == "shift":
                starts[1] += .11
            elif case == "count":
                starts.insert(4, 500.)
            elif case == "late_shift":
                starts[-1] += .02
            elif case == "missing_late":
                starts.pop()
        times, volts = [270.], [-70.]
        for ordinal, start in enumerate(starts):
            peak = -1. if index == 1 and case == "classification" and ordinal == 1 else 20.
            times.extend([start, start + .1, start + .2, start + .3])
            volts.extend([-70., peak, -70., -70.])
        times.append(1270.)
        volts.append(-70.)
        np.savez(tmp_path / (stem + ".npz"), time_ms=times, voltage_mv=volts)
    monkeypatch.setattr(audit, "__file__", str(tmp_path / "audit.py"))
    monkeypatch.setattr("sys.argv", ["audit"] + ([] if factor is None else ["--fine-factor", str(factor)]))
    audit.main()
    suffix = "" if factor is None else f"-{factor}"
    report = json.loads((tmp_path / ("h01-pv-focused-mesh" + suffix + ".json")).read_text())
    assert report["decision"] == expected
    if case == "count":
        assert not report["equal_counts"]
        assert any(report["unmatched_events"].values())
    if case == "classification":
        assert report["equal_counts"] and not report["equal_event_classification"]
