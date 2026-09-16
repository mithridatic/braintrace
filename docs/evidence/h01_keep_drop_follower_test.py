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
