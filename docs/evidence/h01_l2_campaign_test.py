"""Manifest checks, command construction, skip logic, and stage gates of the campaign runner."""
import json
import pytest
from docs.evidence import h01_l2_campaign as campaign


def _manifest(n=2):
    return {"cap": 3, "max_concurrent": 2, "image": "img:1", "cache_dir": ".cache/l2", "library": "lib",
            "output_dir": "h01-l2-campaign", "sweeps": {"50": ["--include-recorded-bias"], "43": []},
            "candidates": [{"name": f"c{i}", "stage": "A", "family": "F1", "settings": {"nseg_factor": 3},
                            "flags": {"kv3_closing_factor": .9}} for i in range(n)]}


def test_manifest_cap_and_unique_names():
    assert campaign.check_manifest(_manifest()) == ["c0", "c1"]
    with pytest.raises(ValueError, match="cap"):
        campaign.check_manifest(_manifest(4))
    duplicate = _manifest()
    duplicate["candidates"][1]["name"] = "c0"
    with pytest.raises(ValueError, match="unique"):
        campaign.check_manifest(duplicate)
    crowded = _manifest()
    crowded["max_concurrent"] = 3
    with pytest.raises(ValueError, match="two concurrent"):
        campaign.check_manifest(crowded)


def test_docker_command_mounts_cache_and_evidence_and_passes_sweep_flags(tmp_path):
    manifest = _manifest()
    command = campaign.docker_command(tmp_path, manifest, manifest["candidates"][0], 50)
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("-w")+1] == "/work/lib"
    assert "--include-recorded-bias" in command and "--sweep" in command
    assert command[-1] == "/evidence/h01-l2-campaign/c0-sweep50"
    assert command[command.index("--candidate-json")+1] == "/evidence/h01-l2-campaign/c0.candidate.json"
    assert campaign.docker_command(tmp_path, manifest, manifest["candidates"][0], 43)[-1].endswith("sweep43")


def test_finished_candidate_with_matching_hash_is_skipped(tmp_path):
    manifest = _manifest()
    candidate = manifest["candidates"][0]
    assert campaign.needs_run(tmp_path, manifest, candidate)
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True)
    text = json.dumps(campaign.candidate_values(candidate), sort_keys=True, indent=2)
    digest = __import__("hashlib").sha256(text.encode()).hexdigest()
    for sweep in (50, 43):
        (folder/f"c0-sweep{sweep}.json").write_text(json.dumps({"candidate_json": {"sha256": digest}}))
    assert not campaign.needs_run(tmp_path, manifest, candidate)
    candidate["flags"]["kv3_closing_factor"] = .8
    assert campaign.needs_run(tmp_path, manifest, candidate)


def test_stage_gates(tmp_path):
    manifest = _manifest()
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True)
    assert campaign.stage_ready(tmp_path, manifest, "0")
    assert not campaign.stage_ready(tmp_path, manifest, "A")
    (folder/"stage0-preservation.json").write_text(json.dumps({"preserved": False}))
    assert not campaign.stage_ready(tmp_path, manifest, "A")
    (folder/"stage0-preservation.json").write_text(json.dumps({"preserved": True}))
    assert campaign.stage_ready(tmp_path, manifest, "A")
    assert not campaign.stage_ready(tmp_path, manifest, "B")


def test_dry_run_writes_flag_file_without_executing(tmp_path):
    manifest = _manifest()
    rows = campaign.run_candidate(tmp_path, manifest, manifest["candidates"][0], dry_run=True)
    assert [r["sweep"] for r in rows] == ["50", "43"] and all(r["returncode"] is None for r in rows)
    written = json.loads((tmp_path/"docs/evidence/h01-l2-campaign/c0.candidate.json").read_text())
    assert written == {"kv3_closing_factor": .9, "nseg_factor": 3}
