"""Tests for the C3 cell table (pure functions; no network)."""

import gzip
import json

import pytest

from docs.evidence.h01_c3_cell_table import (COUNT_COLUMNS, build_table, classify_tags, decode_payload,
                                             load_cell_types, neuron_rows)

VOCAB = load_cell_types()
TAGS = ["L1", "L2", "L3", "WM", "layer-unclassified", "pyramidal", "neuron", "interneuron", "unclassified-neuron",
        "astrocyte", "spiny-stellate", "excitatory/spiny-with-atypical-tree", "bipolar", "possible-interneuron",
        "sparsely-spiny"]


def _info(segments):
    """Segment-properties JSON from ``[(id, tags, counts)]``."""
    props = [{"id": "tags", "type": "tags", "tags": TAGS,
              "values": [[TAGS.index(t) for t in tags] for _, tags, _ in segments]}]
    for k, col in enumerate(COUNT_COLUMNS):
        props.append({"id": col, "type": "number", "values": [c[k] for _, _, c in segments]})
    props.append({"id": "Sp", "type": "number", "values": [0.1 for _ in segments]})
    return {"@type": "neuroglancer_segment_properties",
            "inline": {"ids": [str(i) for i, _, _ in segments], "properties": props}}


def test_load_cell_types_reads_the_runtime_vocabulary_without_the_package():
    assert VOCAB.CLASSES["pyramidal"] == "E" and "L2" in VOCAB.LAYERS and "bipolar" in VOCAB.MODIFIERS


def test_classify_tags_layer_class_and_modifiers():
    assert classify_tags(["L5", "bipolar", "neuron", "pyramidal"], VOCAB) == {
        "layer": "L5", "cell_class": "pyramidal", "modifiers": ["bipolar"]}
    assert classify_tags(["neuron", "spiny-stellate", "L4"], VOCAB)["cell_class"] == "spiny-stellate"
    out = classify_tags(["layer-unclassified", "unclassified-neuron", "possible-interneuron", "neuron"], VOCAB)
    assert out == {"layer": "layer-unclassified", "cell_class": "unclassified-neuron",
                   "modifiers": ["possible-interneuron"]}
    assert classify_tags(["neuron", "dark"], VOCAB) == {"layer": None, "cell_class": None, "modifiers": ["dark"]}


def test_classify_tags_rejects_two_layers_or_two_classes():
    with pytest.raises(ValueError):
        classify_tags(["L1", "L2", "pyramidal", "neuron"], VOCAB)
    with pytest.raises(ValueError):
        classify_tags(["L1", "pyramidal", "interneuron", "neuron"], VOCAB)


def test_decode_payload_handles_gzip_and_plain():
    assert decode_payload(b'{"a": 1}') == '{"a": 1}'
    assert decode_payload(gzip.compress(b'{"a": 1}')) == '{"a": 1}'


def test_neuron_rows_keeps_only_neuron_tagged_segments_with_int_ids_and_counts():
    info = _info([(10, ["L2", "pyramidal", "neuron"], list(range(1, 9))),
                  (11, ["L3", "astrocyte"], [0]*8),
                  (12, ["WM", "interneuron", "neuron", "sparsely-spiny"], [9]*8)])
    rows = neuron_rows(info, VOCAB)
    assert [r["id"] for r in rows] == [10, 12]
    assert rows[0]["NVx"] == 1 and rows[0]["NSp"] == 8 and rows[0]["layer"] == "L2"
    assert rows[1] == {"id": 12, "layer": "WM", "cell_class": "interneuron", "modifiers": ["sparsely-spiny"],
                       **{c: 9 for c in COUNT_COLUMNS}}


def test_build_table_counts_by_class_and_is_compact_json():
    info = _info([(10, ["L2", "pyramidal", "neuron"], [1]*8), (12, ["L2", "pyramidal", "neuron"], [1]*8),
                  (13, ["L2", "neuron"], [1]*8)])
    table = build_table(json.dumps(info), VOCAB, "abc", "2026-09-16T00:00:00Z", source="s")
    assert table["counts_by_class"] == {"None": 1, "pyramidal": 2}
    assert table["neurons"] == 3 and table["sha256"] == "abc" and table["source"] == "s"
    assert json.loads(json.dumps(table, separators=(",", ":")))["rows"][2]["cell_class"] is None
