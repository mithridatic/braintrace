"""Tests for the offline pair merge check (synthetic edge lists; no network)."""

import json

from docs.evidence.h01_pair_merge_check import check, largest_pair, main, render_markdown, verdict


def edge(annotation, pre, post, pre_label="0", post_label="0", type_code=2, within=False):
    return {"annotation_id": annotation, "pre_cell": pre, "post_cell": post, "type": type_code,
            "pre_proofread_label": pre_label, "post_proofread_label": post_label,
            "pre_nearest_offset": None, "post_nearest_offset": None, "endpoints_within_box": within}


def edge_list(edges, cells_a=("1", "2"), cells_b=("3",)):
    return {"edges": edges, "c3_ids_shared_by_cells": {},
            "cells": {"A": {"c3_ids": list(cells_a), "identity_consistent": True},
                      "B": {"c3_ids": list(cells_b), "identity_consistent": True},
                      "C": {"c3_ids": ["9"], "identity_consistent": True}}}


def test_largest_pair_is_unordered():
    edges = [edge("1", "A", "B"), edge("2", "B", "A"), edge("3", "A", "C")]
    assert largest_pair(edges) == (("A", "B"), 2)


def test_undetermined_when_endpoints_are_background_and_no_label_is_shared():
    record = check(edge_list([edge("1", "A", "B"), edge("2", "B", "A", type_code=1)]), {})
    assert record["verdict"] == "suspected merge: undetermined"
    assert record["shared_c3_ids"] == {"edge_list": []} and record["rows_background_both_ends"] == 2
    assert record["direction_type_mix"] == {"A->type2": 1, "B->type1": 1}
    assert record["endpoint_reads"] == {"background/background": 2}


def test_shared_c3_label_in_a_checkpoint_gives_yes():
    record = check(edge_list([edge("1", "A", "B")]), {"ckpt": {"A": {"c3_ids": ["7"]}, "B": {"c3_ids": ["7", "8"]}}})
    assert record["verdict"] == "suspected merge: yes"
    assert record["shared_c3_ids"] == {"edge_list": [], "checkpoint:ckpt": ["7"]}
    assert record["cells_present_in_source"]["checkpoint:ckpt"] == ["A", "B"]


def test_same_nonzero_label_at_both_ends_gives_yes_and_expected_cells_give_no():
    assert check(edge_list([edge("1", "A", "B", "A", "A")]), {})["verdict"] == "suspected merge: yes"
    record = check(edge_list([edge("1", "A", "B", "A", "B", within=True), edge("2", "A", "B")]), {})
    assert record["verdict"] == "suspected merge: no" and record["rows_expected_cells_at_both_ends"] == 1
    assert record["rows"][0]["pre_reads"] == "cell_a" and record["rows"][0]["post_reads"] == "cell_b"
    assert check(edge_list([edge("1", "A", "B", "Z", "0")]), {})["rows"][0]["pre_reads"] == "third_label"
    assert verdict([], []) == "undetermined"


def test_main_writes_json_and_markdown(tmp_path):
    source = tmp_path/"edges.json"
    source.write_text(json.dumps(edge_list([edge("1", "A", "B")])))
    ckpt = tmp_path/"edges-3000.cells.json"
    ckpt.write_text(json.dumps({"key": {}, "cells": {"A": {"c3_ids": ["1"]}}}))
    out = tmp_path/"merge.json"
    record = main(["--edge-list", str(source), "--checkpoint", str(ckpt), str(tmp_path/"missing.json"), "--output", str(out)])
    saved = json.loads(out.read_text())
    assert saved["verdict"] == record["verdict"] == "suspected merge: undetermined"
    assert saved["sources"]["checkpoints"] == [str(ckpt)] and saved["cells_present_in_source"]["checkpoint:edges-3000.cells.json"] == ["A"]
    text = (tmp_path/"merge.md").read_text()
    assert "**Verdict: suspected merge: undetermined**" in text and "| 1 | A | B | 2 |" in text
    assert render_markdown(record).startswith("# Merge check: pair A / B")
