"""Per-input skip, untested-resume, evaluation counting, abort override, and dry-run plan of the runner."""

import json

import pytest

from docs.evidence import h01_campaign as campaign
from docs.evidence.h01_campaign_test import _manifest


def _ungated(manifest):
    """Move every candidate to the ungated stage so ``main`` gets past the gate."""
    for candidate in manifest["candidates"]:
        candidate["stage"] = "0"
    return manifest


def _finish(tmp_path, manifest, candidate, input_name, sha=None):
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    record = {"candidate_json": {"sha256": sha or campaign.flag_hash(candidate)}}
    (folder/f"{candidate['name']}-{input_name}.json").write_text(json.dumps(record))


def test_input_finished_is_per_input_and_hash_bound(tmp_path):
    manifest = _manifest()
    candidate = manifest["candidates"][0]
    assert not campaign.input_finished(tmp_path, manifest, candidate, "019")
    _finish(tmp_path, manifest, candidate, "019")
    assert campaign.input_finished(tmp_path, manifest, candidate, "019")
    assert not campaign.input_finished(tmp_path, manifest, candidate, "027")
    _finish(tmp_path, manifest, candidate, "027", sha="0"*64)
    assert not campaign.input_finished(tmp_path, manifest, candidate, "027")
    assert campaign.needs_run(tmp_path, manifest, candidate)


def test_legacy_flat_log_is_grouped_into_evaluations():
    rows = [{"name": "c0", "input": "019", "seconds": 1., "returncode": 0, "aborted": False, "command": []},
            {"name": "c0", "input": "027", "seconds": 2., "returncode": None, "aborted": True, "command": []},
            {"name": "c1", "input": "019", "seconds": 3., "returncode": 1, "aborted": False, "command": []}]
    log = campaign.upgrade_log(rows, _manifest())
    assert [e["evaluation_index"] for e in log] == [1, 2]
    assert log[0]["inputs"] == {"019": "completed", "027": "untested"}
    assert log[1]["inputs"] == {"019": "failed"}
    assert log[0]["candidate_sha256"] == campaign.flag_hash(_manifest()["candidates"][0])
    assert len(log[0]["runs"]) == 2 and log[0]["abort_seconds"] == 900
    assert campaign.upgrade_log(log, _manifest()) == log


def test_plan_skips_finished_inputs_and_resumes_untested_ones(tmp_path):
    manifest = _manifest()
    candidate = manifest["candidates"][0]
    assert campaign.plan_inputs(tmp_path, manifest, candidate, [], None) == {"019": "run", "027": "run"}
    _finish(tmp_path, manifest, candidate, "019")
    entry = campaign.register_evaluation([], manifest, candidate, 900, None)
    entry["inputs"]["027"] = "untested"
    plan = campaign.plan_inputs(tmp_path, manifest, candidate, [entry], None)
    assert plan == {"019": "skip", "027": "resume"}
    assert campaign.plan_inputs(tmp_path, manifest, candidate, [entry], ["027"]) == {"027": "resume"}
    with pytest.raises(ValueError, match="not an input"):
        campaign.plan_inputs(tmp_path, manifest, candidate, [entry], ["099"])
    candidate["flags"]["sodium_h_slope_mv"] = 6.
    assert campaign.plan_inputs(tmp_path, manifest, candidate, [entry], None) == {"019": "run", "027": "run"}


def test_evaluation_counts_once_and_resume_reuses_the_entry():
    manifest = _manifest()
    manifest["prior_evaluations"] = 1
    log = []
    first = campaign.register_evaluation(log, manifest, manifest["candidates"][0], 900, None)
    assert first["evaluation_index"] == 2 and first["inputs"] == {"019": "untested", "027": "untested"}
    again = campaign.register_evaluation(log, manifest, manifest["candidates"][0], 900, None)
    assert again is first and len(log) == 1
    second = campaign.register_evaluation(log, manifest, manifest["candidates"][1], 900, None)
    assert second["evaluation_index"] == 3 and len(log) == 2
    assert campaign.evaluations_used(log, manifest) == 3
    with pytest.raises(ValueError, match="cap"):
        campaign.register_evaluation(log, manifest, {"name": "c9", "flags": {}, "stage": "A"}, 900, None)


def test_rows_update_input_status_without_touching_the_index():
    manifest = _manifest()
    entry = campaign.register_evaluation([], manifest, manifest["candidates"][0], 900, None)
    campaign.record_rows(entry, [{"name": "c0", "input": "019", "seconds": 1., "returncode": 0, "aborted": False, "command": []},
                                 {"name": "c0", "input": "027", "seconds": 2., "returncode": None, "aborted": True, "command": []}])
    assert entry["inputs"] == {"019": "completed", "027": "untested"} and entry["evaluation_index"] == 1
    campaign.record_rows(entry, [{"name": "c0", "input": "027", "seconds": 3., "returncode": 2, "aborted": False, "command": []}])
    assert entry["inputs"]["027"] == "failed" and len(entry["runs"]) == 3 and entry["evaluation_index"] == 1


def test_abort_override_above_manifest_needs_a_registered_reason():
    manifest = _manifest()
    assert campaign.abort_limit(manifest, None, None) == 900
    assert campaign.abort_limit(manifest, 600, None) == 600
    with pytest.raises(ValueError, match="abort_override_reason"):
        campaign.abort_limit(manifest, 1200, None)
    with pytest.raises(ValueError, match="abort_override_reason"):
        campaign.abort_limit(manifest, 1200, "  ")
    assert campaign.abort_limit(manifest, 1200, "shared host, measured 2.3x") == 1200
    entry = campaign.register_evaluation([], manifest, manifest["candidates"][0], 1200, "shared host")
    assert entry["abort_seconds"] == 1200 and entry["abort_override_reason"] == "shared host"


def test_run_candidate_runs_only_the_requested_inputs_under_the_given_abort(tmp_path, monkeypatch):
    seen = []

    class Done:
        returncode = 0

    def fake(command, **kwargs):
        seen.append((command[-1], kwargs.get("timeout")))
        return Done()
    monkeypatch.setattr(campaign.subprocess, "run", fake)
    manifest = _manifest()
    rows = campaign.run_candidate(tmp_path, manifest, manifest["candidates"][0], False, inputs=["027"], abort_seconds=1200)
    assert [r["input"] for r in rows] == ["027"] and rows[0]["returncode"] == 0
    assert seen == [("/evidence/h01-i-campaign/c0-027", 1200)]


def test_main_dry_run_prints_the_plan_and_writes_no_log(tmp_path, monkeypatch, capsys):
    manifest = _ungated(_manifest())
    candidate = manifest["candidates"][0]
    _finish(tmp_path, manifest, candidate, "019")
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    (folder/"campaign-log.json").write_text(json.dumps(
        [{"name": "c0", "input": "019", "seconds": 1., "returncode": 0, "aborted": False, "command": []},
         {"name": "c0", "input": "027", "seconds": 2., "returncode": None, "aborted": True, "command": []}]))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(campaign, "repo_root", lambda: tmp_path)
    campaign.main(["--manifest", str(tmp_path/"manifest.json"), "--stage", "0", "--dry-run"])
    out = capsys.readouterr().out
    assert "c0 019 skip" in out and "c0 027 resume evaluation 1" in out
    assert "c1 019 run evaluation 2" in out and "c1 027 run evaluation 2" in out
    assert "evaluations used 2 of 3" in out
    assert json.loads((folder/"campaign-log.json").read_text())[0]["input"] == "019"


def test_main_resume_completes_untested_inputs_without_a_new_evaluation(tmp_path, monkeypatch, capsys):
    class Done:
        returncode = 0
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs: Done())
    manifest = _ungated(_manifest(1))
    candidate = manifest["candidates"][0]
    _finish(tmp_path, manifest, candidate, "019")
    folder = tmp_path/"docs/evidence"/manifest["output_dir"]
    (folder/"campaign-log.json").write_text(json.dumps(
        [{"name": "c0", "input": "019", "seconds": 1., "returncode": 0, "aborted": False, "command": []},
         {"name": "c0", "input": "027", "seconds": 2., "returncode": None, "aborted": True, "command": []}]))
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(campaign, "repo_root", lambda: tmp_path)
    with pytest.raises(SystemExit, match="abort_override_reason"):
        campaign.main(["--manifest", str(tmp_path/"manifest.json"), "--stage", "0", "--abort-seconds", "1200"])
    campaign.main(["--manifest", str(tmp_path/"manifest.json"), "--stage", "0", "--only-input", "027",
                   "--abort-seconds", "1200", "--abort-override-reason", "shared host"])
    log = json.loads((folder/"campaign-log.json").read_text())
    assert len(log) == 1 and log[0]["evaluation_index"] == 1
    assert log[0]["inputs"] == {"019": "completed", "027": "completed"}
    assert log[0]["abort_seconds"] == 1200 and log[0]["abort_override_reason"] == "shared host"
    assert [r["input"] for r in log[0]["runs"]] == ["019", "027", "027"]
    assert "c0 027 resume evaluation 1" in capsys.readouterr().out


def test_flag_hash_equals_the_sha_the_driver_records_for_the_written_file(tmp_path):
    import hashlib
    from pathlib import Path

    manifest = _manifest()
    candidate = manifest["candidates"][0]
    campaign.run_candidate(tmp_path, manifest, candidate, dry_run=True)
    written = (tmp_path/"docs/evidence/h01-i-campaign/c0.candidate.json").read_bytes()
    assert hashlib.sha256(written).hexdigest() == campaign.flag_hash(candidate)
    evidence = Path(campaign.__file__).resolve().parent
    real = json.loads((evidence/"h01-e-gain-manifest.json").read_text())
    report = json.loads((evidence/"h01-e-gain/g0-b3-sweep43.json").read_text())
    assert campaign.flag_hash(real["candidates"][0]) == report["candidate_json"]["sha256"]
    assert campaign.input_finished(evidence.parents[1], real, real["candidates"][0], "sweep43")
