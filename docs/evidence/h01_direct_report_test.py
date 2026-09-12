"""End-to-end artifact checks for direct observations and visual-review gates."""

import hashlib
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from docs.evidence.h01_direct_currents_test import capacitor
from docs.evidence.h01_direct_report import main, write_bundle


def files(tmp_path, geometry=True, command=True):
    data, report = capacitor()
    data["time_ms"] = 1000+data["time_ms"]*1300
    report["sweep"] = 56
    report["candidate_json"] = {"sha256": "test-candidate"}
    report["soma_observation_units"]["soma_cai_mm"] = "mM; local"
    data["soma_cai_mm"] = np.linspace(.0001, .0002, len(data["time_ms"]))
    stem = tmp_path/"model"
    np.savez(stem.with_suffix(".npz"), **data)
    stem.with_suffix(".json").write_text(json.dumps(report if geometry else {}))
    human = {"time_ms": data["time_ms"], "corrected_voltage_mv": data["voltage_mv"]}
    if command:
        human["total_current_na"] = data["applied_current_na"]
    hp = tmp_path/"human.npz"
    np.savez(hp, **human)
    return stem, hp


def test_artifacts_retain_sources_pairs_states_and_pending_visual_review(tmp_path):
    stem, human = files(tmp_path)
    out = write_bundle([stem], tmp_path/"direct", [human])
    record = json.loads((tmp_path/"direct.json").read_text())
    assert record["visual_review"] == "pending" and record["causal_verdict"] == "not_established"
    assert record["records"][1]["observation"]["current_evidence"] == "command_only"
    with np.load(out["arrays"]) as selected:
        assert any("state_soma_cai_mm" in k for k in selected.files)
        assert any("charge_storage_pc" in k for k in selected.files)
    assert hashlib.sha256((tmp_path/"direct.npz").read_bytes()).hexdigest() == record["selected_samples"]["sha256"]
    assert len(out["plots"]) == 4
    for path in out["plots"]:
        with open(path, "rb") as stream:
            assert stream.read(8) == b"\x89PNG\r\n\x1a\n"


def test_missing_current_record_renders_unavailability_not_fabricated_pairs(tmp_path):
    stem, human = files(tmp_path, geometry=False, command=False)
    result = write_bundle([stem], tmp_path/"direct", [human])
    assert len(result["plots"]) == 4
    record = json.loads((tmp_path/"direct.json").read_text())
    assert all(r["observation"]["current_evidence"] == "unavailable" for r in record["records"])


def test_compute_only_cli_still_requires_later_visual_review(tmp_path, capsys):
    stem, human = files(tmp_path)
    main(["--run", str(stem), "--human", str(human), "--output", str(tmp_path/"direct"), "--no-plots"])
    result = json.loads(capsys.readouterr().out)
    assert result["plots"] == [] and result["visual_review"] == "pending"
    with pytest.raises(ValueError):
        write_bundle([], tmp_path/"empty")
