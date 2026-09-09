"""Search tree: every node cites an existing decision record, the tree is connected, files are written."""

import json

import h01_search_tree as tree


def test_every_reference_exists():
    missing = [ref for _, _, _, _, ref in tree.NODES if not (tree.EVIDENCE/ref).exists()]
    assert missing == []


def test_every_node_has_a_known_status_and_a_parent_in_the_tree():
    ids = {n[0] for n in tree.NODES}
    assert all(n[3] in tree.STYLE for n in tree.NODES)
    assert all(n[1] in ids for n in tree.NODES if n[1] is not None)
    assert sum(n[1] is None for n in tree.NODES) == 1


def test_layout_places_every_node_and_keeps_parents_above_children():
    pos = tree.layout()
    assert set(pos) == {n[0] for n in tree.NODES}
    for node_id, parent, *_ in tree.NODES:
        if parent:
            assert pos[parent][1] > pos[node_id][1]


def test_main_writes_figure_and_summary(tmp_path):
    counts = tree.main(tmp_path)
    assert (tmp_path/"h01-search-tree.png").exists() and (tmp_path/"h01-search-tree.svg").exists()
    summary = json.loads((tmp_path/"h01-search-tree.json").read_text())
    assert summary["status_counts"] == counts and counts["eliminated"] >= 4 and counts["fail"] == 2
