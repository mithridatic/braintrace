import pytest

from . import h01_cell_types as module


def test_cell_type_reads_layer_class_and_modifiers():
    kind = module.cell_type(["L5", "bipolar", "neuron", "pyramidal"])
    assert kind == module.H01CellType("L5", "pyramidal", ("bipolar",), "E")
    assert module.cell_type(["L1", "interneuron", "neuron", "web-like-interneuron"]).polarity == "I"
    assert module.cell_type(["L6", "excitatory/spiny-with-atypical-tree", "neuron"]).cell_class.startswith("excitatory")


@pytest.mark.parametrize("tags", [["neuron", "pyramidal"], ["L2", "neuron"], ["L2", "L3", "pyramidal"],
                                  ["L2", "pyramidal", "interneuron"]])
def test_cell_type_rejects_missing_or_ambiguous_tags(tags):
    with pytest.raises(ValueError):
        module.cell_type(tags)


def test_donor_match_names_each_mismatch():
    assert module.donor_match(module.cell_type(["L2", "pyramidal", "neuron"]))["match"] == "matched"
    assert module.donor_match(module.cell_type(["L4", "pyramidal", "neuron"]))["match"] == "layer"
    assert module.donor_match(module.cell_type(["L2", "excitatory/spiny-with-atypical-tree", "neuron"]))["match"] == "class"
    assert module.donor_match(module.cell_type(["L4", "pyramidal", "sparsely-spiny", "neuron"]))["match"] == "layer, modifier"
    plain = module.donor_match(module.cell_type(["L5", "interneuron", "neuron"]))
    assert plain["match"] == "matched" and "parvalbumin donor is assumed" in plain["note"]
    assert module.donor_match(module.cell_type(["L3", "interneuron", "neuron"]))["donor"] == "h01-pv-regional-mesh-axon2187"


def test_type_rows_keep_cell_ids_and_lists():
    rows = module.type_rows([{"cell_id": "1", "tags": ["L2", "pyramidal", "neuron"]},
                             {"cell_id": "2", "tags": ["L5", "interneuron", "sparsely-spiny", "neuron"]}])
    assert [r["cell_id"] for r in rows] == ["1", "2"]
    assert rows[1]["modifiers"] == ["sparsely-spiny"] and rows[1]["match"] == "modifier"
