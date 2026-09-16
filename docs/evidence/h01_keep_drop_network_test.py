"""Tests for the kept-cell topology/components filter."""

import json
import sys

import pytest

from docs.evidence import h01_keep_drop_network as network


def _node(i, cell, polarity):
    return dict(index=i, cell_id=cell, tags=["L2"], polarity=polarity, dale_sign=1 if polarity == "E" else -1)


def _contact(annotation, pre, post, nodes, ready=True):
    index = {n["cell_id"]: n["index"] for n in nodes}
    return dict(annotation_id=annotation, pre_cell=pre, post_cell=post, pre_index=index[pre], post_index=index[post],
                construction_ready=ready, blockers=[])


def _toy():
    nodes = [_node(0, "10", "E"), _node(1, "20", "E"), _node(2, "30", "I"), _node(3, "40", "E")]
    contacts = [_contact("a", "10", "20", nodes), _contact("b", "20", "30", nodes), _contact("c", "30", "40", nodes, False)]
    topology = dict(nodes=nodes, contacts=contacts, excluded_contacts=[], archive_sha256="x", scope="s",
                    max_distance_um=1., counts=dict(nodes=4, excitatory=3, inhibitory=1, verified_contacts=3,
                                                    construction_ready=2))
    cells = [dict(cell_id=c, components=1, nodes=10, largest_component=0, largest_share=1., largest_has_soma=True)
             for c in ("10", "20", "30", "40")]
    comp_contacts = [dict(annotation_id="a", side="pre", cell_id="10"), dict(annotation_id="a", side="post", cell_id="20"),
                     dict(annotation_id="b", side="pre", cell_id="20"), dict(annotation_id="b", side="post", cell_id="30")]
    components = dict(archive="z", summary=dict(cells=4, components=4), cells=cells, contacts=comp_contacts)
    return topology, components


def test_keeping_two_nodes_leaves_only_their_contact_and_reindexes():
    topology, components = _toy()
    new_topology, new_components, report = network.filter_network(topology, components, ["20", "30"])
    assert [n["cell_id"] for n in new_topology["nodes"]] == ["20", "30"]
    assert [n["index"] for n in new_topology["nodes"]] == [0, 1]
    assert [c["annotation_id"] for c in new_topology["contacts"]] == ["b"]
    assert (new_topology["contacts"][0]["pre_index"], new_topology["contacts"][0]["post_index"]) == (0, 1)
    assert new_topology["counts"] == dict(nodes=2, excitatory=1, inhibitory=1, verified_contacts=1, construction_ready=1)
    assert new_topology["archive_sha256"] == "x" and new_topology["max_distance_um"] == 1.
    assert [r["cell_id"] for r in new_components["cells"]] == ["20", "30"]
    assert [c["cell_id"] for c in new_components["contacts"]] == ["20", "20", "30"]
    assert new_components["summary"] == dict(cells=2, components=4) and new_components["archive"] == "z"
    assert report == dict(kept=2, nodes_before=4, nodes_after=2, contacts_before=3, contacts_after=1,
                          dropped_contacts=[("10", "20"), ("30", "40")], components_before=4, components_after=2)
    assert len(topology["nodes"]) == 4 and topology["nodes"][1]["index"] == 1   # inputs untouched


def test_unknown_kept_id_raises():
    topology, components = _toy()
    with pytest.raises(ValueError, match="99"):
        network.filter_network(topology, components, ["10", "99"])


def test_all_kept_is_identity():
    topology, components = _toy()
    new_topology, new_components, report = network.filter_network(topology, components, ["10", "20", "30", "40"])
    assert new_topology == topology and new_components == components
    assert report["dropped_contacts"] == [] and report["contacts_after"] == 3


def test_cli_writes_both_files(tmp_path, capsys):
    topology, components = _toy()
    paths = {name: tmp_path/(name+".json") for name in ("topology", "components", "decision", "out-t", "out-c")}
    paths["topology"].write_text(json.dumps(topology))
    paths["components"].write_text(json.dumps(components))
    paths["decision"].write_text(json.dumps(dict(kept=["10", "20"])))
    argv = sys.argv
    sys.argv = ["x", "--topology", str(paths["topology"]), "--components", str(paths["components"]),
                "--decision", str(paths["decision"]), "--out-topology", str(paths["out-t"]),
                "--out-components", str(paths["out-c"])]
    try:
        network.main()
    finally:
        sys.argv = argv
    written_topology = json.loads(paths["out-t"].read_text())
    written_components = json.loads(paths["out-c"].read_text())
    assert [n["cell_id"] for n in written_topology["nodes"]] == ["10", "20"]
    assert [c["annotation_id"] for c in written_topology["contacts"]] == ["a"]
    assert len(written_components["cells"]) == 2
    provenance = written_topology["filtered_from"]
    assert provenance == written_components["filtered_from"]
    assert all(len(provenance[k]) == 64 for k in ("topology_sha256", "components_sha256", "decision_sha256"))
    assert json.loads(capsys.readouterr().out.strip())["kept"] == 2
