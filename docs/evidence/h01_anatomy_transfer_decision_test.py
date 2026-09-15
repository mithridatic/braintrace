"""Tests for the anatomy-transfer decision assembler."""

import json

from docs.evidence import h01_anatomy_transfer_decision as decision


def _run(count, human, repeats=None, finite=True, donor_fit=None):
    verdict = decision.__dict__  # noqa: F841 - keep module reference explicit for readers
    difference = abs(count-human)
    exact = "held" if difference == 0 else ("missed_within_one" if difference <= 1 else "rejected")
    band = None if not repeats else ("held" if min(repeats) <= count <= max(repeats) else "missed")
    return dict(cell="c", current_na=.1, registered_human_count=human, repeat_counts=repeats,
                donor_model_count_on_own_anatomy=donor_fit, rest_before_pulse_mv=-80., peak_mv=20.,
                output_site=dict(count=count, finite=finite), soma_site=dict(count=count, finite=finite),
                verdict_vs_human=dict(exact=exact, repeat_band=band))


def test_donor_verdict_rules():
    assert decision.donor_verdict(_run(13, 12), _run(13, 12))["passed"] is True
    assert decision.donor_verdict(_run(14, 12), _run(14, 12))["passed"] is False
    assert decision.donor_verdict(_run(13, 14, (14, 14, 13, 12)), _run(13, 14, (14, 14, 13, 12)))["passed"] is True
    assert decision.donor_verdict(_run(11, 14, (14, 14, 13, 12)), _run(11, 14, (14, 14, 13, 12)))["passed"] is False
    assert decision.donor_verdict(_run(12, 12), None)["passed"] is False           # no dt-half confirmation
    assert decision.donor_verdict(_run(12, 12), _run(11, 12))["passed"] is False   # dt-half disagrees
    assert decision.donor_verdict(_run(12, 12, finite=False), _run(12, 12))["passed"] is False


def test_decide_fraction_and_hashes(tmp_path):
    for donor, label in decision.PRIMARY.items():
        folder = tmp_path/label
        folder.mkdir()
        good = label == "transfer-l4-090"
        (folder/"run.json").write_text(json.dumps(_run(12 if good else 2, 12)))
        half = tmp_path/(label+"-dthalf")
        half.mkdir()
        (half/"run.json").write_text(json.dumps(_run(12 if good else 2, 12)))
    result = decision.decide(tmp_path)
    assert result["anatomy_transfer"] == .25 and result["measured"] is True
    assert result["donors"]["l4-pyramidal-allen-527952884"]["passed"] is True
    assert sum(row["passed"] for row in result["donors"].values()) == 1
    assert len(result["input_hashes"]) == 8 and all(len(h) == 64 for h in result["input_hashes"].values())
