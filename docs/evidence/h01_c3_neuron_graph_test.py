"""Tests for the C3 neuron-graph streamer (pure functions and file plumbing; no network, no fastavro)."""

import gzip
import json

import pytest

from docs.evidence.h01_c3_neuron_graph import (BASE_URL, COLUMNS, curl_command, keep_row, merge_parts, new_progress,
                                               part_path, pending_shards, shard_index, totals, write_json)


def _record(pre, post, **kw):
    rec = {"pre_synaptic_site": {"neuron_id": pre, "class_label": "AXON"},
           "post_synaptic_partner": {"neuron_id": post, "class_label": "DENDRITE"},
           "type": 2, "location": {"x": 10, "y": 20, "z": 3}, "confidence": 0.98765}
    rec.update(kw)
    return rec


def test_keep_row_requires_both_ends_in_the_neuron_set_and_rounds_confidence():
    ids = {1, 2}
    assert keep_row(_record(1, 2), ids, 7) == [1, 2, 2, 10, 20, 3, "AXON", "DENDRITE", 0.9877, 7]
    assert keep_row(_record(1, 3), ids, 7) is None
    assert keep_row(_record(3, 2), ids, 7) is None
    assert keep_row(_record("1", "2"), ids, 0)[:2] == [1, 2]
    assert keep_row(_record(None, 2), ids, 0) is None
    assert keep_row(_record(1, 2, confidence=None), ids, 0)[8] is None
    assert len(COLUMNS) == len(keep_row(_record(1, 2), ids, 0))


def test_shard_index_parses_export_names_only():
    assert shard_index("export000000000165") == 165 and shard_index("export000000000000") == 0
    with pytest.raises(ValueError):
        shard_index("export000000000000.json")


def test_totals_sums_receipts_and_counts_shards():
    entries = [{"bytes": 10, "records": 5, "kept": 1, "download_seconds": 1.5, "scan_seconds": 2.0},
               {"bytes": 20, "records": 7, "kept": 0, "download_seconds": 0.5, "scan_seconds": 1.0}]
    assert totals(entries) == {"bytes": 30, "records": 12, "kept": 1, "download_seconds": 2.0,
                               "scan_seconds": 3.0, "shards": 2}
    assert totals([])["shards"] == 0


def test_pending_shards_resumes_only_receipted_shards_whose_part_exists(tmp_path):
    listing = {"export000000000001": 1, "export000000000000": 1, "export000000000002": 1}
    part_path("export000000000000", tmp_path).write_text("[]")
    progress = {"shards": [{"id": "export000000000000"}, {"id": "export000000000002"}]}
    assert pending_shards(listing, progress, tmp_path) == ["export000000000001", "export000000000002"]
    assert [e["id"] for e in progress["shards"]] == ["export000000000000"]


def test_merge_parts_concatenates_in_shard_order_as_columns_plus_rows(tmp_path):
    part_path("export000000000000", tmp_path).write_text(json.dumps([[1, 2, 2, 0, 0, 0, "A", "D", 0.5, 0]]))
    part_path("export000000000001", tmp_path).write_text(json.dumps([]))
    part_path("export000000000002", tmp_path).write_text(json.dumps([[3, 4, 1, 1, 1, 1, "A", "S", 0.9, 2]]))
    out = tmp_path / "g.json.gz"
    receipt = merge_parts(["export000000000000", "export000000000001", "export000000000002"], out, tmp_path)
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    assert doc["columns"] == list(COLUMNS)
    assert doc["rows"] == [[1, 2, 2, 0, 0, 0, "A", "D", 0.5, 0], [3, 4, 1, 1, 1, 1, "A", "S", 0.9, 2]]
    assert receipt["rows"] == 2 and receipt["bytes"] == out.stat().st_size and len(receipt["sha256"]) == 64


def test_write_json_is_atomic_and_new_progress_carries_the_layout(tmp_path):
    target = tmp_path / "p.json"
    write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1} and not target.with_suffix(".tmp").exists()
    progress = new_progress({"export000000000000": 1}, 6, {"x": "h"})
    assert progress["columns"] == list(COLUMNS) and progress["shards_total"] == 1 and progress["workers"] == 6
    assert progress["shards"] == [] and progress["stopped_reason"] is None


def test_curl_command_fails_on_http_errors_aborts_stalls_and_caps_wall_time(tmp_path):
    argv = curl_command("export000000000005", tmp_path / "x.part")
    assert argv[0] == "curl" and "--fail" in argv and argv[-1] == BASE_URL + "export000000000005"
    assert "--max-time" in argv and "--speed-limit" in argv and "--speed-time" in argv
    assert argv[argv.index("-o") + 1] == str(tmp_path / "x.part")
