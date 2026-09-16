"""Tests for the kept-partner graph streamer (ownership join, keep rule, receipts; no network, no fastavro)."""

import gzip
import json

import pytest

from docs.evidence.h01_c3_kept_partner_graph import (COLUMNS, KINDS, classify, keep_row, kind_totals,
                                                     load_kept_owners, new_progress, side_owner)
from docs.evidence.h01_c3_neuron_graph import merge_parts, part_path

AUDIT = [{"record": "a.json", "existing_spatial_owners": ["11"], "c3_ids": [21]},
         {"record": "b.json", "existing_spatial_owners": ["12"], "c3_ids": [22]},
         {"record": "c.json", "existing_spatial_owners": ["13"], "c3_ids": [23]},
         {"record": "d.json", "existing_spatial_owners": [], "c3_ids": []}]
MEMBERSHIP = [{"file": "a.json", "regions": {"axon": [1, 2], "dendrite": ["3"], "unknown": []}},
              {"file": "b.json", "regions": {"dendrite": [4]}},
              {"file": "c.json", "regions": {"dendrite": [5]}},
              {"file": "d.json", "regions": {"dendrite": [6]}}]
CTX = {"base_owner": {1: 11, 2: 11, 3: 11, 4: 12}, "c3_owner": {21: 11, 22: 12}, "neuron_ids": frozenset({21, 22, 30, 31})}


def _record(pre_c3, post_c3, pre_base, post_base, pre_class="AXON", **kw):
    rec = {"pre_synaptic_site": {"neuron_id": pre_c3, "base_neuron_id": pre_base, "class_label": pre_class},
           "post_synaptic_partner": {"neuron_id": post_c3, "base_neuron_id": post_base, "class_label": "DENDRITE"},
           "type": 2, "location": {"x": 10, "y": 20, "z": 3}, "confidence": 0.98765}
    rec.update(kw)
    return rec


def test_load_kept_owners_restricts_to_the_kept_cells_and_normalises_ids():
    base_owner, c3_owner = load_kept_owners(AUDIT, MEMBERSHIP, kept=(11, "12"))
    assert base_owner == {1: 11, 2: 11, 3: 11, 4: 12} and c3_owner == {21: 11, 22: 12}


def test_load_kept_owners_fails_on_a_shared_base_segment_or_a_missing_membership():
    shared = MEMBERSHIP + [{"file": "b.json", "regions": {"axon": [1]}}]
    with pytest.raises(ValueError, match="claimed by"):
        load_kept_owners(AUDIT, shared, kept=(11, 12))
    with pytest.raises(ValueError, match="without membership"):
        load_kept_owners(AUDIT, MEMBERSHIP[:1], kept=(11, 12))


def test_side_owner_prefers_base_ownership_then_the_kept_c3_id():
    assert side_owner({"neuron_id": None, "base_neuron_id": 1}, CTX) == 11
    assert side_owner({"neuron_id": 22, "base_neuron_id": 99}, CTX) == 12
    assert side_owner({"neuron_id": "21", "base_neuron_id": "4"}, CTX) == 12
    assert side_owner({"neuron_id": 30, "base_neuron_id": 99}, CTX) is None
    assert side_owner({"neuron_id": None, "base_neuron_id": None}, CTX) is None


def test_classify_orders_the_kinds_and_rejects_untabulated_partners():
    ids = CTX["neuron_ids"]
    assert classify(11, 12, None, 22, ids) == "kept_to_kept"
    assert classify(11, 11, 21, 21, ids) == "same_cell"
    assert classify(11, None, None, 30, ids) == "kept_to_tabulated"
    assert classify(None, 12, 30, None, ids) == "tabulated_to_kept"
    assert classify(11, None, None, 99, ids) is None
    assert classify(None, 12, None, None, ids) is None
    assert classify(None, None, 30, 31, ids) is None


def test_keep_row_emits_the_column_layout_per_kind():
    kind, row = keep_row(_record(None, 30, 1, 99), 5, CTX, 7)
    assert kind == "kept_to_tabulated"
    assert row == [11, None, None, 30, 2, 10, 20, 3, "AXON", "DENDRITE", 0.9877, 7, 5] and len(row) == len(COLUMNS)
    assert keep_row(_record("31", None, 99, "4", pre_class="DENDRITE"), 0, CTX, 0)[0] == "tabulated_to_kept"
    # a kept cell's own C3 id on the post side is kept -> kept, never a tabulated partner
    assert keep_row(_record(None, 22, 1, 99), 0, CTX, 0)[0] == "kept_to_kept"
    assert keep_row(_record(21, 21, 1, 2), 0, CTX, 0)[0] == "same_cell"
    assert keep_row(_record(30, 31, 98, 99), 0, CTX, 0) is None
    assert keep_row(_record(None, 99, 1, 99), 0, CTX, 0) == ("untabulated_partner", None)
    assert keep_row(_record(None, None, 99, 4), 0, CTX, 0) == ("untabulated_partner", None)
    assert keep_row(_record(None, 30, 1, 99, confidence=None), 0, CTX, 0)[1][10] is None


def test_kind_totals_sums_receipts_over_every_kind():
    entries = [{"kinds": {"kept_to_kept": 1, "kept_to_tabulated": 4}, "axon_pre": {"kept_to_tabulated": 3}},
               {"kinds": {"same_cell": 2}, "axon_pre": {}}, {}]
    out = kind_totals(entries)
    assert out["rows"] == {"kept_to_kept": 1, "same_cell": 2, "kept_to_tabulated": 4, "tabulated_to_kept": 0}
    assert out["axon_pre_rows"] == {"kept_to_kept": 0, "same_cell": 0, "kept_to_tabulated": 3, "tabulated_to_kept": 0}
    assert set(out["rows"]) == set(KINDS)
    entries[0]["dropped"] = {"pre_kept": 8, "post_kept": 181}
    assert kind_totals(entries)["dropped_untabulated_partner"] == {"pre_kept": 8, "post_kept": 181}


def test_new_progress_carries_the_layout_and_merge_uses_the_thirteen_columns(tmp_path):
    progress = new_progress({"export000000000000": 1}, 6, {"x": "h"})
    assert progress["columns"] == list(COLUMNS) and progress["kinds"] == list(KINDS)
    assert progress["shards"] == [] and len(progress["kept_released_ids"]) == 17
    row = [11, None, None, 30, 2, 10, 20, 3, "AXON", "DENDRITE", 0.9877, 0, 5]
    part_path("export000000000000", tmp_path).write_text(json.dumps([row]))
    out = tmp_path / "g.json.gz"
    receipt = merge_parts(["export000000000000"], out, tmp_path, columns=COLUMNS)
    doc = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    assert doc["columns"] == list(COLUMNS) and doc["rows"] == [row] and receipt["rows"] == 1
