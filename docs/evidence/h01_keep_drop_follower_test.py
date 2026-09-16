"""Tests for the dt-half follower's pure selection logic."""

from docs.evidence import h01_keep_drop_follower as follower


def test_pending_skips_candidates_already_launched(tmp_path):
    (tmp_path/"transfer-all-b-dthalf").mkdir()
    (tmp_path/"transfer-all-b-dthalf"/"launch.json").write_text("{}")
    donors = {"a": "l4-pyramidal-allen-527952884", "b": "l2-pyramidal-allen-541563728"}
    assert follower.pending(tmp_path, ["a", "b"], donors) == [("a", "l4-pyramidal-allen-527952884")]
    assert follower.pending(tmp_path, [], donors) == []


def test_chains_done_needs_every_done_file(tmp_path):
    for k in (1, 2, 3):
        (tmp_path/f"keep-chain-{k}.done").write_text("x")
    assert follower.chains_done(tmp_path) is False
    (tmp_path/"keep-chain-4.done").write_text("x")
    assert follower.chains_done(tmp_path) is True


def test_dt_half_line_resolves_the_runner_and_carries_the_extra_args():
    line = follower.dt_half_line("693197378", "l2-pyramidal-allen-541563728", "docs/evidence/h01-c3-keep-drop/run_transfer.sh",
                                 "--archive .cache/h01/c3-candidates-20260916.zip --cell-table .cache/h01/c3-segment-properties.json")
    assert line.startswith("docs/evidence/h01-c3-keep-drop/run_transfer.sh transfer-all-693197378-dthalf 3600 --")
    assert "--dt-ms 0.0025" in line and line.endswith("--cell-table .cache/h01/c3-segment-properties.json")
    assert "$R" not in line
    assert follower.dt_half_line("1", "l4-pyramidal-allen-527952884", "var/h01-driven/run_transfer.sh").endswith("--dt-ms 0.0025")


def test_parse_args_defaults_and_c3_overrides():
    default = follower.parse_args([])
    assert default.gate == "bands" and default.chains == 4 and default.runner == "var/h01-driven/run_transfer.sh"
    c3 = follower.parse_args(["--folder", "var/c3-keep", "--gate", "failure", "--chains", "2",
                              "--runner", "docs/evidence/h01-c3-keep-drop/run_transfer.sh", "--extra-args", "--archive x"])
    assert c3.gate == "failure" and c3.chains == 2 and c3.extra_args == "--archive x"
    assert c3.runner == "docs/evidence/h01-c3-keep-drop/run_transfer.sh"
    assert follower.parse_args(["--folder", "var/c3-keep"]).runner == "var/c3-keep/run_transfer.sh"


def test_main_waits_for_the_ramp_under_the_failure_gate_and_launches_once(tmp_path, monkeypatch):
    import json
    import subprocess
    from docs.evidence.h01_anatomy_transfer_decision_test import _cell_run_failure, _ramp, AFTER, REST
    l4 = "l4-pyramidal-allen-527952884"
    folder = tmp_path/"runs"
    (folder/"transfer-all-200").mkdir(parents=True)
    (folder/"transfer-all-200"/"run.json").write_text(json.dumps(_cell_run_failure(cell="200", count=12)))
    types, rests, datums = tmp_path/"types.json", tmp_path/"rests.json", tmp_path/"datums.json"
    types.write_text(json.dumps(dict(rows=[dict(cell_id="200", donor_key=l4)])))
    rests.write_text(json.dumps(dict(donors={l4: dict(rest_mv=REST-.4, sd_mv=.02)})))
    datums.write_text(json.dumps(dict(donors={l4: dict(rheobase_pa=50., sweep_step_pa=20., highest_firing_pa=170.,
                                                        repeat_counts=[12], rest_repeat_mean_mv=REST, rest_repeat_sd_mv=.1,
                                                        after_repeat_mean_mv=AFTER, after_repeat_sd_mv=.2)})))
    launched = []
    def fake_run(cmd, **kwargs):
        launched.append(cmd[2])
        (folder/"transfer-all-200-dthalf").mkdir()
        (folder/"transfer-all-200-dthalf"/"launch.json").write_text("{}")
    monkeypatch.setattr(subprocess, "run", fake_run)
    sleeps = []
    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:   # the ramp lands after the first poll
            (folder/"transfer-all-200-ramp").mkdir()
            (folder/"transfer-all-200-ramp"/"run.json").write_text(json.dumps(dict(cell="200", **_ramp())))
        (folder/"keep-chain-1.done").write_text("CHAIN-DONE")
        (folder/"keep-chain-2.done").write_text("CHAIN-DONE")
    monkeypatch.setattr(follower.time, "sleep", fake_sleep)
    output = tmp_path/"primary.json"
    follower.main(["--folder", str(folder), "--types", str(types), "--rests", str(rests), "--datums", str(datums),
                   "--gate", "failure", "--chains", "2", "--output", str(output), "--poll", "1",
                   "--runner", "R.sh", "--extra-args", "--archive z.zip"])
    assert len(launched) == 1 and launched[0].startswith("R.sh transfer-all-200-dthalf 3600 --")
    assert launched[0].endswith("--dt-ms 0.0025 --archive z.zip")
    written = json.loads(output.read_text())
    assert written["gate"] == "failure" and written["candidates"] == ["200"]
    assert len(sleeps) == 2   # poll 1: pending ramp; poll 2: launched, chains done -> exit
