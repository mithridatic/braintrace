"""Tests for the C3 partner ranking (synthetic graph and table; no network)."""

import gzip
import json

import pytest

from docs.evidence.h01_c3_cell_table import load_cell_types
from docs.evidence.h01_c3_partner_rank import (build, donor_fields, induced_subgraph, kept_c3_ids, load_graph,
                                               markdown, partner_counts, polarity_of, proofread_ids,
                                               rank_candidates, summarize)

VOCAB = load_cell_types()


def _cell(cid, layer, cls, nsi, mods=()):
    return {"id": cid, "layer": layer, "cell_class": cls, "modifiers": list(mods), "NVx": 1, "NSO": 1, "NSI": nsi,
            "NSIe": 0, "NSIi": 0, "NDe": 0, "NAx": 0, "NSp": 0}


def _row(pre, post, typ=2):
    return {"pre": pre, "post": post, "type": typ, "x": 0, "y": 0, "z": 0, "pre_class": "AXON",
            "post_class": "DENDRITE", "confidence": 0.9, "shard": 0}


AUDIT = [{"record": "a", "existing_spatial_owners": ["1684504313"], "c3_ids": [1684504313]},
         {"record": "b", "existing_spatial_owners": ["1669770671"], "c3_ids": [1931553044]},
         {"record": "c", "existing_spatial_owners": ["999"], "c3_ids": [888]},
         {"record": "d", "existing_spatial_owners": [], "c3_ids": []}]
POP = [{"cell_id": "1684504313"}, {"cell_id": "1669770671"}, {"cell_id": "999"}]


def test_kept_c3_ids_follows_the_audit_and_fails_loudly_on_a_missing_row():
    kept = kept_c3_ids(AUDIT, released=(1669770671, 1684504313))
    assert kept == [{"released_id": 1669770671, "c3_id": 1931553044}, {"released_id": 1684504313, "c3_id": 1684504313}]
    with pytest.raises(KeyError):
        kept_c3_ids(AUDIT, released=(5,))


def test_proofread_ids_unions_released_and_c3_ids():
    assert proofread_ids(AUDIT, POP) == {1684504313, 1669770671, 1931553044, 999, 888}


def test_partner_counts_separates_directions_and_kept_internal_rows():
    rows = [_row(5, 1), _row(5, 1), _row(1, 6), _row(1, 2), _row(7, 8)]
    partners, per_kept, internal = partner_counts(rows, {1, 2})
    assert partners == {5: {"to_kept": 2, "from_kept": 0}, 6: {"to_kept": 0, "from_kept": 1}}
    assert internal == 1 and per_kept[1]["synapses_in"] == 2 and per_kept[1]["synapses_out"] == 2
    assert per_kept[1]["partners"] == {5, 6} and per_kept[1]["kept_partners"] == {2}
    assert per_kept[2]["partners"] == set() and per_kept[2]["synapses_in"] == 1


def test_polarity_of_applies_the_eligibility_rule():
    assert polarity_of(_cell(1, "L2", "pyramidal", 1), VOCAB) == "E"
    assert polarity_of(_cell(1, "WM", "spiny-stellate", 1), VOCAB) == "E"
    assert polarity_of(_cell(1, "L3", "interneuron", 1), VOCAB) == "I"
    assert polarity_of(_cell(1, "layer-unclassified", "pyramidal", 1), VOCAB) is None
    assert polarity_of(_cell(1, None, "interneuron", 1), VOCAB) is None
    assert polarity_of(_cell(1, "L2", "pyramidal", 1, ["possible-interneuron"]), VOCAB) is None
    assert polarity_of(_cell(1, "L2", "unclassified-neuron", 1), VOCAB) is None


def test_donor_fields_resolves_the_registry_and_reports_mismatches():
    assert donor_fields(_cell(1, "L4", "pyramidal", 1), "E", VOCAB) == {
        "donor_key": "l4-pyramidal-allen-527952884", "donor_profile": "h01-l4-allen-527952884", "donor_match": "matched"}
    out = donor_fields(_cell(1, "L5", "spiny-stellate", 1, ["bipolar", "dark"]), "E", VOCAB)
    assert out["donor_key"] == "l2-pyramidal-allen-541563728" and out["donor_match"] == "layer, class, modifier"
    assert donor_fields(_cell(1, "L3", "interneuron", 1), "I", VOCAB)["donor_key"] == "l3-sst-interneuron-hl5mn1"
    assert donor_fields(_cell(1, "L6", "interneuron", 1), "I", VOCAB)["donor_match"] == "layer"


def test_rank_candidates_orders_by_total_then_nsi_and_excludes_proofread_and_unknown_cells():
    partners = {5: {"to_kept": 2, "from_kept": 0}, 6: {"to_kept": 1, "from_kept": 1}, 7: {"to_kept": 1, "from_kept": 0},
                8: {"to_kept": 9, "from_kept": 0}, 9: {"to_kept": 3, "from_kept": 0}, 10: {"to_kept": 1, "from_kept": 0}}
    table = {5: _cell(5, "L2", "pyramidal", 10), 6: _cell(6, "L2", "pyramidal", 50), 7: _cell(7, "L3", "interneuron", 1),
             8: _cell(8, "L3", "pyramidal", 1), 9: _cell(9, "L1", "interneuron", 2)}
    cands, eligible = rank_candidates(partners, table, {8}, VOCAB, top_n=1)
    assert eligible == {"E": 2, "I": 2}
    assert [(c["c3_id"], c["polarity"]) for c in cands] == [(6, "E"), (9, "I")]
    cands, _ = rank_candidates(partners, table, set(), VOCAB, top_n=5)
    assert [c["c3_id"] for c in cands] == [8, 6, 5, 9, 7]
    assert cands[1]["synapses_total"] == 2 and cands[1]["nsi"] == 50 and cands[1]["donor_match"] == "matched"


def test_build_summary_markdown_and_subgraph(tmp_path):
    rows = [_row(5, 1684504313), _row(5, 1684504313), _row(1931553044, 6), _row(1684504313, 1931553044), _row(7, 8),
            _row(5, 6), _row(888, 1684504313), _row(888, 1684504313)]
    table = [_cell(5, "L2", "pyramidal", 10), _cell(6, "L3", "interneuron", 3), _cell(7, "L2", "pyramidal", 1),
             _cell(888, "L2", "pyramidal", 1)]
    doc = build(rows, table, AUDIT[:3], POP, VOCAB, released=(1669770671, 1684504313))
    s = doc["summary"]
    assert [k["c3_id"] for k in doc["kept"]] == [1931553044, 1684504313]
    assert s["partners_total"] == 3 and s["partners_ge2_to_kept"] == 2 and s["kept_internal_rows"] == 1
    assert s["partners_excluded_as_proofread"] == [888]
    assert [c["c3_id"] for c in doc["candidates"]] == [5, 6]
    assert s["prediction"]["verdict"] == "refuted" and s["prediction"]["falsifier_fewer_than_20_partners"]
    assert s["prediction"]["every_kept_has_partner"] is True and s["per_kept_cell_partner_counts"]["1931553044"]["partners"] == 1
    assert {(r["pre"], r["post"]) for r in doc["subgraph"]} == {(5, 1684504313), (1931553044, 6), (1684504313, 1931553044), (5, 6)}
    text = markdown({"summary": s, "candidates": doc["candidates"]})
    assert "| 1 | 5 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 2 | 0 | 10 |" in text
    gz = tmp_path / "g.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        json.dump({"columns": ["pre", "post"], "rows": [[1, 2]]}, f)
    assert load_graph(gz) == [{"pre": 1, "post": 2}]
    assert induced_subgraph([{"pre": 1, "post": 2}, {"pre": 1, "post": 3}], {1, 2}) == [{"pre": 1, "post": 2}]


def test_summarize_supports_when_every_kept_has_a_partner_and_forty_partners_send_two():
    partners = {i: {"to_kept": 2, "from_kept": 0} for i in range(40)}
    per_kept = {1: {"synapses_in": 80, "synapses_out": 0, "partners": set(range(40)), "kept_partners": set()}}
    kept = [{"released_id": 1, "c3_id": 1}]
    s = summarize(partners, per_kept, 0, kept, [], {"E": 0, "I": 0}, [], 80)
    assert s["prediction"]["verdict"] == "supported" and s["partners_ge2_either_direction"] == 40
    partners[0]["to_kept"] = 1
    assert summarize(partners, per_kept, 0, kept, [], {"E": 0, "I": 0}, [], 80)["prediction"]["verdict"] == "refuted"
