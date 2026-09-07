import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h01_population_types as types  # noqa: E402


def test_donor_notes_read_registered_donor_decisions_only(tmp_path):
    folder = tmp_path/"h01-donors"
    folder.mkdir()
    key = next(iter(types.DONORS))
    (folder/"stage-a-decision.json").write_text(json.dumps(
        {"donor_key": key, "reproduction_status": "rejected", "population_label": "type-matched, x"}), encoding="utf-8")
    (folder/"stage-b-decision.json").write_text(json.dumps({"donor_key": "not-a-donor"}), encoding="utf-8")
    notes = types.donor_notes(tmp_path)
    assert notes == {key: {"status": "rejected", "label": "type-matched, x", "decision": "h01-donors/stage-a-decision.json"}}
    assert types.donor_notes(tmp_path/"missing") == {}


def test_render_lists_donor_notes_between_donors_and_match_states():
    key = next(iter(types.DONORS))
    report = {"summary": {"cells": 1, "matched": 1, "by_match": {"matched": 1}, "by_type": []},
              "donor_notes": {key: {"status": "rejected", "label": "L", "decision": "h01-donors/d.json"}}, "rows": []}
    text = types.render(report)
    assert "## Donor physiology notes" in text
    assert f"| {key} | rejected | L | [h01-donors/d.json](h01-donors/d.json) |" in text
    assert text.index("## Donors") < text.index("## Donor physiology notes") < text.index("## Match states")
    assert "## Donor physiology notes" not in types.render({**report, "donor_notes": {}})
