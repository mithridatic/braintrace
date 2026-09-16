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


def _row(pre, post, typ=2, pre_class="AXON", pre_owner=None, post_owner=None):
    """Kept-partner graph row; kept ends are given by released id in ``pre_owner`` / ``post_owner``."""
    return {"pre_owner": pre_owner, "post_owner": post_owner, "pre_c3": pre, "post_c3": post, "type": typ,
            "x": 0, "y": 0, "z": 0, "pre_class": pre_class, "post_class": "DENDRITE", "confidence": 0.9,
            "shard": 0, "index": 0}


def _counts(to_kept, from_kept, to_axon=None, from_axon=None):
    return {"to_kept": to_kept, "from_kept": from_kept, "to_kept_axon": to_kept if to_axon is None else to_axon,
            "from_kept_axon": from_kept if from_axon is None else from_axon}


KEPT = {1669770671: 1931553044, 1684504313: 1684504313}


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


def test_partner_counts_separates_directions_axon_pre_and_kept_internal_rows():
    a, b = 1669770671, 1684504313
    rows = [_row(5, 1931553044, post_owner=a), _row(5, None, post_owner=a),
            _row(5, None, post_owner=a, pre_class="DENDRITE"),
            _row(None, 6, pre_owner=a), _row(None, 7, pre_owner=a, pre_class="SOMA"),
            _row(None, None, pre_owner=a, post_owner=b),
            _row(None, None, pre_owner=a, post_owner=b, pre_class="DENDRITE"),
            _row(1931553044, 1931553044, pre_owner=a, post_owner=a)]
    partners, per_kept, internal = partner_counts(rows, KEPT)
    assert partners == {5: {"to_kept": 3, "from_kept": 0, "to_kept_axon": 2, "from_kept_axon": 0},
                        6: {"to_kept": 0, "from_kept": 1, "to_kept_axon": 0, "from_kept_axon": 1},
                        7: {"to_kept": 0, "from_kept": 1, "to_kept_axon": 0, "from_kept_axon": 0}}
    assert internal == {"rows": 2, "axon_rows": {(1931553044, 1684504313): 1}, "dendrite_pre_rows_ignored": 1,
                        "same_cell_rows": 1}
    k = per_kept[1931553044]
    assert k["partners"] == {5, 6} and k["partners_any_class"] == {5, 6, 7} and k["kept_partners"] == {1684504313}
    assert (k["synapses_in"], k["synapses_in_axon"], k["synapses_out"], k["synapses_out_axon"]) == (3, 2, 4, 2)
    assert per_kept[1684504313]["synapses_in"] == 2 and per_kept[1684504313]["synapses_in_axon"] == 1
    assert per_kept[1684504313]["partners"] == set()


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


def test_rank_candidates_orders_by_axon_total_then_nsi_and_excludes_proofread_unknown_and_axonless_cells():
    partners = {5: _counts(2, 0), 6: _counts(1, 1), 7: _counts(1, 0), 8: _counts(9, 0), 9: _counts(3, 0),
                10: _counts(1, 0), 11: _counts(40, 0, to_axon=0)}
    table = {5: _cell(5, "L2", "pyramidal", 10), 6: _cell(6, "L2", "pyramidal", 50), 7: _cell(7, "L3", "interneuron", 1),
             8: _cell(8, "L3", "pyramidal", 1), 9: _cell(9, "L1", "interneuron", 2), 11: _cell(11, "L2", "pyramidal", 99)}
    cands, eligible = rank_candidates(partners, table, {8}, VOCAB, top_n=1)
    assert eligible == {"E": 2, "I": 2}
    assert [(c["c3_id"], c["polarity"]) for c in cands] == [(6, "E"), (9, "I")]
    cands, _ = rank_candidates(partners, table, set(), VOCAB, top_n=5)
    assert [c["c3_id"] for c in cands] == [8, 6, 5, 9, 7]
    assert cands[1]["synapses_total"] == 2 and cands[1]["nsi"] == 50 and cands[1]["donor_match"] == "matched"
    assert cands[1]["synapses_total_axon"] == 2 and cands[1]["dendrite_pre_rows_ignored"] == 0
    partners[6]["to_kept"] = 5
    cands, _ = rank_candidates(partners, table, set(), VOCAB, top_n=5)
    assert cands[1]["synapses_to_kept"] == 5 and cands[1]["synapses_to_kept_axon"] == 1
    assert cands[1]["dendrite_pre_rows_ignored"] == 4


def test_build_summary_markdown_and_subgraph(tmp_path):
    a, b = 1669770671, 1684504313
    rows = [_row(5, 1684504313, post_owner=b), _row(5, None, post_owner=b), _row(None, 6, pre_owner=a),
            _row(None, None, pre_owner=b, post_owner=a), _row(None, 7, pre_owner=a, pre_class="DENDRITE"),
            _row(888, None, post_owner=b), _row(888, None, post_owner=b), _row(None, None, pre_owner=a, post_owner=a)]
    table = [_cell(5, "L2", "pyramidal", 10), _cell(6, "L3", "interneuron", 3), _cell(7, "L2", "pyramidal", 1),
             _cell(888, "L2", "pyramidal", 1)]
    doc = build(rows, table, AUDIT[:3], POP, VOCAB, released=(a, b))
    s = doc["summary"]
    assert [k["c3_id"] for k in doc["kept"]] == [1931553044, 1684504313]
    assert s["graph_source"] == "kept-partner" and s["graph_rows"] == 8
    assert s["partners_total"] == 3 and s["partners_any_class_total"] == 4 and s["partners_ge2_to_kept"] == 2
    assert s["partners_ge2_either_direction"] == 2 and s["kept_internal_rows"] == 1 and s["kept_same_cell_rows"] == 1
    assert s["kept_internal_axon_rows"] == [{"pre": 1684504313, "post": 1931553044, "count": 1}]
    assert s["kept_internal_dendrite_pre_rows_ignored"] == 0
    assert s["partners_excluded_as_proofread"] == [888]
    assert s["row_labels"]["pre_class"] == {"AXON": 7, "DENDRITE": 1}
    assert [c["c3_id"] for c in doc["candidates"]] == [5, 6]
    assert s["prediction"]["verdict"] == "refuted" and s["prediction"]["verdict_either_direction"] == "refuted"
    assert s["prediction"]["falsifier_fewer_than_20_partners"]
    per = s["per_kept_cell_partner_counts"]["1931553044"]
    assert s["prediction"]["every_kept_has_partner"] is True and per["partners"] == 1 and per["partners_any_class"] == 2
    assert per["kept_partners"] == [1684504313] and per["synapses_out"] == 2 and per["synapses_out_axon"] == 1
    assert {(r["pre_c3"], r["post_c3"], r["pre_owner"], r["post_owner"]) for r in doc["subgraph"]} == {
        (5, 1684504313, None, b), (5, None, None, b), (None, 6, a, None), (None, None, b, a), (None, None, a, a)}
    text = markdown({"summary": s, "candidates": doc["candidates"]})
    assert "| 1 | 5 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 2 | 0 | 0 | 10 |" in text
    assert "1684504313 -> 1931553044 x1" in text and "C3-id-only" in text
    gz = tmp_path / "g.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        json.dump({"columns": ["pre_c3", "post_c3"], "rows": [[1, 2]]}, f)
    assert load_graph(gz) == [{"pre_c3": 1, "post_c3": 2}]
    assert induced_subgraph([_row(1, 2), _row(1, 3), _row(None, 2, pre_owner=a)], {1, 2, 1931553044}, KEPT) == [
        _row(1, 2), _row(None, 2, pre_owner=a)]


def test_summarize_supports_when_every_kept_has_a_partner_and_forty_partners_send_two_axon_synapses():
    partners = {i: _counts(2, 0) for i in range(40)}
    per_kept = {1: {"synapses_in": 80, "synapses_out": 0, "synapses_in_axon": 80, "synapses_out_axon": 0,
                    "partners": set(range(40)), "partners_any_class": set(range(40)), "kept_partners": set()}}
    internal = {"rows": 0, "axon_rows": {}, "dendrite_pre_rows_ignored": 0, "same_cell_rows": 0}
    kept = [{"released_id": 1, "c3_id": 1}]
    s = summarize(partners, per_kept, internal, kept, [], {"E": 0, "I": 0}, [], 80)
    assert s["prediction"]["verdict"] == "supported" and s["partners_ge2_either_direction"] == 40
    assert s["prediction"]["verdict_either_direction"] == "supported"
    partners[0]["to_kept_axon"] = 1
    s = summarize(partners, per_kept, internal, kept, [], {"E": 0, "I": 0}, [], 80)
    assert s["prediction"]["verdict"] == "refuted" and s["partners_ge2_to_kept"] == 39
    partners[0]["from_kept_axon"] = 1  # the either-direction reading still counts it
    s = summarize(partners, per_kept, internal, kept, [], {"E": 0, "I": 0}, [], 80)
    assert s["prediction"]["verdict"] == "refuted" and s["prediction"]["verdict_either_direction"] == "supported"
    partners[100] = _counts(3, 0, to_axon=0)  # a dendrite-pre-only partner is not a partner
    s = summarize(partners, per_kept, internal, kept, [], {"E": 0, "I": 0}, [], 80)
    assert s["partners_total"] == 40 and s["partners_any_class_total"] == 41
