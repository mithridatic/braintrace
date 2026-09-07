"""Manifest rules, command construction, skip logic, stage gates, and abort of the generic runner."""

import json

import pytest

from docs.evidence import h01_campaign as campaign


def _manifest(n=2):
    return {"cap": 3, "max_concurrent": 2, "abort_seconds": 900, "image": "img:1", "cache_dir": ".cache/pv",
            "library": "lib", "driver": "h01_pv_neuron_reference.py", "output_dir": "h01-i-campaign",
            "inputs": {"019": ["--current-na", "0.19", "--bias-na", "0.0314"], "027": ["--current-na", "0.27"]},
            "stages": {"A": {"gate": "stage0-decision.json", "prediction": "p", "rejection": "r", "unchanged": "u"},
                       "0": {"prediction": "p", "rejection": "r", "unchanged": "u"}},
            "candidates": [{"name": f"c{i}", "stage": "A", "settings": {"nseg_factor": 9},
                            "flags": {"sodium_h_slope_mv": 5.}} for i in range(n)]}


def test_manifest_rules():
    assert campaign.check_manifest(_manifest()) == ["c0", "c1"]
    with pytest.raises(ValueError, match="cap"):
        campaign.check_manifest(_manifest(4))
    prior = _manifest(2)
    prior["prior_evaluations"] = 2
    with pytest.raises(ValueError, match="cap"):
        campaign.check_manifest(prior)
    duplicate = _manifest()
    duplicate["candidates"][1]["name"] = "c0"
    with pytest.raises(ValueError, match="unique"):
        campaign.check_manifest(duplicate)
    crowded = _manifest()
    crowded["max_concurrent"] = 3
    with pytest.raises(ValueError, match="two concurrent"):
        campaign.check_manifest(crowded)
    no_abort = _manifest()
    no_abort["abort_seconds"] = 0
    with pytest.raises(ValueError, match="abort_seconds"):
        campaign.check_manifest(no_abort)
    unregistered = _manifest()
    unregistered["stages"]["A"].pop("prediction")
    with pytest.raises(ValueError, match="pre-registered prediction"):
        campaign.check_manifest(unregistered)
    unknown = _manifest()
    unknown["candidates"][0]["stage"] = "Z"
    with pytest.raises(ValueError, match="does not define"):
        campaign.check_manifest(unknown)


def test_docker_command_uses_manifest_driver_and_input_flags(tmp_path):
    manifest = _manifest()
    command = campaign.docker_command(tmp_path, manifest, manifest["candidates"][0], "019")
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("-w")+1] == "/work/lib"
    assert command[command.index("python")+1] == "/evidence/h01_pv_neuron_reference.py"
    assert command[command.index("--candidate-json")+1] == "/evidence/h01-i-campaign/c0.candidate.json"
    assert "--bias-na" in command and command[-1] == "/evidence/h01-i-campaign/c0-019"


def test_finished_candidate_with_matching_hash_is_skipped(tmp_path):
    manifest = _manifest()
    candidate = manifest["candidates"][0]
    assert campaign.needs_run(tmp_path, manifest, candidate)
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True)
    for name in manifest["inputs"]:
        (folder/f"c0-{name}.json").write_text(json.dumps({"candidate_json": {"sha256": campaign.flag_hash(candidate)}}))
    assert not campaign.needs_run(tmp_path, manifest, candidate)
    candidate["flags"]["sodium_h_slope_mv"] = 6.
    assert campaign.needs_run(tmp_path, manifest, candidate)


def test_stage_gate_needs_a_recorded_decision(tmp_path):
    manifest = _manifest()
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True)
    assert campaign.stage_ready(tmp_path, manifest, "0")
    assert not campaign.stage_ready(tmp_path, manifest, "A")
    (folder/"stage0-decision.json").write_text(json.dumps({"decision": ""}))
    assert not campaign.stage_ready(tmp_path, manifest, "A")
    (folder/"stage0-decision.json").write_text(json.dumps({"decision": "rank at nseg 9"}))
    assert campaign.stage_ready(tmp_path, manifest, "A")


def test_dry_run_writes_flag_file_and_records_no_return_code(tmp_path):
    manifest = _manifest()
    rows = campaign.run_candidate(tmp_path, manifest, manifest["candidates"][0], dry_run=True)
    assert [r["input"] for r in rows] == ["019", "027"]
    assert all(r["returncode"] is None and not r["aborted"] for r in rows)
    written = json.loads((tmp_path/"docs/evidence/h01-i-campaign/c0.candidate.json").read_text())
    assert written == {"nseg_factor": 9, "sodium_h_slope_mv": 5.}


def test_abort_is_recorded_when_the_container_exceeds_the_limit(tmp_path, monkeypatch):
    import subprocess

    killed = []

    def slow(command, **kwargs):
        if command[:2] == ["docker", "kill"]:
            killed.append(command[2])
            return None
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(campaign.subprocess, "run", slow)
    manifest = _manifest()
    manifest["abort_seconds"] = 1
    rows = campaign.run_candidate(tmp_path, manifest, manifest["candidates"][0], dry_run=False)
    assert all(r["aborted"] and r["returncode"] is None for r in rows)
    assert killed == ["h01-i-campaign-c0-019", "h01-i-campaign-c0-027"]
    command = campaign.docker_command(tmp_path, manifest, manifest["candidates"][0], "019")
    assert command[command.index("--name")+1] == "h01-i-campaign-c0-019"


def test_container_stderr_is_persisted_beside_the_report(tmp_path, monkeypatch):
    import subprocess

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 139, stdout="", stderr="Segmentation fault" + chr(10))
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    manifest = _manifest()
    rows = campaign.run_candidate(tmp_path, manifest, manifest["candidates"][0], dry_run=False)
    assert [r["returncode"] for r in rows] == [139, 139]
    folder = tmp_path/"docs/evidence/h01-i-campaign"
    assert (folder/"c0-019.stderr.txt").read_text(encoding="utf-8") == "Segmentation fault" + chr(10)
    assert (folder/"c0-027.stderr.txt").exists()
    campaign.persist_stderr(folder, {"name": "c0"}, "z", None)
    assert not (folder/"c0-z.stderr.txt").exists()
    campaign.persist_stderr(folder, {"name": "c0"}, "z", b"bytes")
    assert (folder/"c0-z.stderr.txt").read_text(encoding="utf-8") == "bytes"
