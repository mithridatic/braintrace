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
    assert plain["match"] == "matched" and "parvalbumin basket donor is assumed" in plain["note"]
    sst = module.donor_match(module.cell_type(["L3", "interneuron", "neuron"]))
    assert sst["donor"] == "h01-sst-l3-hl5mn1" and sst["match"] == "matched"
    assert "putative SST" in sst["note"] and "parvalbumin" not in sst["note"]
    assert module.donor_match(module.cell_type(["L2", "interneuron", "neuron"]))["donor"] == "h01-pv-regional-mesh-axon2187"


def test_type_rows_keep_cell_ids_and_lists():
    rows = module.type_rows([{"cell_id": "1", "tags": ["L2", "pyramidal", "neuron"]},
                             {"cell_id": "2", "tags": ["L5", "interneuron", "sparsely-spiny", "neuron"]}])
    assert [r["cell_id"] for r in rows] == ["1", "2"]
    assert rows[1]["modifiers"] == ["sparsely-spiny"] and rows[1]["match"] == "modifier"


E_KEY, I_KEY = "l2-pyramidal-allen-541563728", "l5-pv-basket-hl5bn1"
SST_KEY = "l3-sst-interneuron-hl5mn1"


def test_registry_keys_and_polarity_view():
    assert set(module.DONORS) == {E_KEY, I_KEY, SST_KEY}
    assert module.DEFAULT_DONOR_KEYS == {"E": E_KEY, "I": I_KEY}
    assert module.DONORS_BY_POLARITY == {"E": module.DONORS[E_KEY], "I": module.DONORS[I_KEY]}
    e, i = module.DONORS[E_KEY], module.DONORS[I_KEY]
    assert (e["channel_prefix"], e["sodium_reversal_mv"], e["potassium_reversal_mv"]) == ("H01L2", 53., -107.)
    assert (i["channel_prefix"], i["sodium_reversal_mv"], i["potassium_reversal_mv"]) == ("H01PV", 50., -85.)
    assert e["modifiers"] == () and i["polarity"] == "I"


def test_donor_for_falls_back_to_polarity_default():
    assert module.donor_for(module.cell_type(["L2", "pyramidal", "neuron"])) == E_KEY
    assert module.donor_for(module.cell_type(["L4", "pyramidal", "sparsely-spiny", "neuron"])) == E_KEY
    assert module.donor_for(module.cell_type(["L6", "excitatory/spiny-with-atypical-tree", "neuron"])) == E_KEY
    assert module.donor_for(module.cell_type(["L1", "interneuron", "bipolar", "neuron"])) == I_KEY


def test_donor_for_precedence_with_extended_registry(monkeypatch):
    def donor(layer, cell_class, modifiers, polarity):
        return dict(module.DONORS[E_KEY], layer=layer, cell_class=cell_class, modifiers=modifiers, polarity=polarity)
    extended = dict(module.DONORS)
    extended["l4-pyramidal-x"] = donor("L4", "pyramidal", (), "E")
    extended["l4-pyramidal-sparse-x"] = donor("L4", "pyramidal", ("sparsely-spiny",), "E")
    extended["l6-atypical-x"] = donor("L6", "excitatory/spiny-with-atypical-tree", (), "E")
    extended["l3-bipolar-int-x"] = donor("L3", "interneuron", ("bipolar",), "I")
    monkeypatch.setattr(module, "DONORS", extended)
    assert module.donor_for(module.cell_type(["L4", "pyramidal", "sparsely-spiny", "neuron"])) == "l4-pyramidal-sparse-x"
    assert module.donor_for(module.cell_type(["L4", "pyramidal", "bipolar", "neuron"])) == "l4-pyramidal-x"
    assert module.donor_for(module.cell_type(["L4", "pyramidal", "neuron"])) == "l4-pyramidal-x"
    assert module.donor_for(module.cell_type(["L2", "excitatory/spiny-with-atypical-tree", "neuron"])) == "l6-atypical-x"
    assert module.donor_for(module.cell_type(["L3", "interneuron", "neuron"])) == SST_KEY
    assert module.donor_for(module.cell_type(["L3", "interneuron", "bipolar", "neuron"])) == "l3-bipolar-int-x"
    assert module.donor_for(module.cell_type(["L5", "pyramidal", "neuron"])) == E_KEY
    assert module.DEFAULT_DONOR_KEYS == {"E": E_KEY, "I": I_KEY}


def test_donor_for_tags_tolerates_partial_tags():
    assert module.donor_for_tags(("pyramidal",), "E") == E_KEY
    assert module.donor_for_tags((), "I") == I_KEY
    assert module.donor_for_tags(("L2", "pyramidal", "neuron"), "E") == E_KEY
    with pytest.raises(KeyError):
        module.donor_for_tags((), "X")


def test_donor_match_reports_donor_key_with_unchanged_shape():
    row = module.donor_match(module.cell_type(["L4", "pyramidal", "neuron"]))
    assert set(row) == {"donor", "donor_key", "match", "note"}
    assert row["donor_key"] == E_KEY and row["donor"] == "h01-l2-kv3-ninety-ca133" and row["match"] == "layer"
    rows = module.type_rows([{"cell_id": "9", "tags": ["L5", "interneuron", "neuron"]}])
    assert rows[0]["donor_key"] == I_KEY


def test_sst_donor_resolves_l3_interneurons_only():
    """The HL5MN1 record takes L3 interneurons without modifiers; every other interneuron stays on HL5BN1."""
    sst = module.DONORS[SST_KEY]
    assert (sst["layer"], sst["cell_class"], sst["modifiers"], sst["polarity"]) == ("L3", "interneuron", (), "I")
    assert (sst["channel_prefix"], sst["sodium_reversal_mv"], sst["potassium_reversal_mv"]) == ("H01PV", 50., -85.)
    assert sst["subtype"] == "unknown (putative SST)"
    assert module.donor_for(module.cell_type(["L3", "interneuron", "neuron"])) == SST_KEY
    # Rule 2 of the precedence: a (layer, class) donor without modifiers takes the modified L3 type too.
    assert module.donor_for(module.cell_type(["L3", "interneuron", "bipolar", "neuron"])) == SST_KEY
    assert module.donor_match(module.cell_type(["L3", "interneuron", "bipolar", "neuron"]))["match"] == "modifier"
    for layer in ("L1", "L2", "L4", "L5", "L6"):
        assert module.donor_for(module.cell_type([layer, "interneuron", "neuron"])) == I_KEY
    assert module.donor_for_tags(("L3", "interneuron", "neuron"), "I") == SST_KEY
    assert module.DEFAULT_DONOR_KEYS["I"] == I_KEY and module.DONORS_BY_POLARITY["I"] is module.DONORS[I_KEY]
