"""Offline identity, sign, and cable-placement checks for H01 connectivity."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from . import h01_connectivity as module
from .h01_anatomy_test import imported


@pytest.fixture
def annotations():
    tags = {"12": ("pyramidal",), "13": ("interneuron",),
            "14": ("excitatory/spiny-with-atypical-tree",)}
    return SimpleNamespace(select=lambda: tuple(tags), metadata=lambda identity: SimpleNamespace(tags=tags[identity]))


def edge(pre="12", post="13", identity="1", kind=2):
    result = dict(annotation_id=identity, pre_cell=pre, post_cell=post,
                  type=kind, endpoints_verified=True)
    for side, cell in (("pre", pre), ("post", post)):
        result.update({side+"_voxel": [800, 0, 0], side+"_proofread_label": cell,
                       side+"_nearest_offset": [0, 0, 0]})
    return result


def report(edges=None):
    return dict(archive_sha256=module.ARCHIVE_SHA256,
                cells={str(i): dict(identity_consistent=True) for i in (12, 13, 14)},
                edges=[edge()] if edges is None else edges)


def test_inventory_signs_and_duplicate_contacts(annotations):
    source = report([edge(), edge(), edge("13", "12", "2", 1)])
    result = module.select_connectivity(source, annotations)
    assert [n["dale_sign"] for n in result["nodes"]] == [1, -1, 1]
    assert len(result["nodes"]) == 3  # Retain the isolated neuron.
    assert [e["connection_type"] for e in result["contacts"]] == ["EI", "IE"]
    assert result["excluded_contacts"][0]["reason"] == "duplicate annotation"
    result["contacts"][0]["pre_voxel"][0] = 1
    assert source["edges"][0]["pre_voxel"][0] == 800


@pytest.mark.parametrize("tags", [(), ("neuron",), ("pyramidal", "interneuron")])
def test_unknown_or_conflicting_cell_types(tags):
    with pytest.raises(ValueError, match="unambiguous"):
        module.cell_sign(tags)


@pytest.mark.parametrize("field,value", [
    ("endpoints_verified", False), ("pre_proofread_label", "14"),
    ("post_nearest_offset", [0, 0, 1]), ("pre_cell", "999"),
    ("type", 1), ("pre_voxel", [np.nan, 0, 0]), ("pre_voxel", [-1, 0, 0]),
    ("pre_voxel", [1, 0]),
])
def test_unsupported_contacts_are_excluded(annotations, field, value):
    item = edge()
    item[field] = value
    result = module.select_connectivity(report([item]), annotations)
    assert result["contacts"] == [] and len(result["excluded_contacts"]) == 1


def test_inconsistent_cell_audit_and_conflicting_duplicate(annotations):
    source = report()
    source["cells"]["12"]["identity_consistent"] = False
    assert not module.select_connectivity(source, annotations)["contacts"]
    source = report([edge(), edge(post="14")])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        module.select_connectivity(source, annotations)
    source["archive_sha256"] = "wrong"
    with pytest.raises(ValueError, match="archive"):
        module.select_connectivity(source, annotations)


def test_nearest_component_uses_cable_not_nearest_node(tmp_path):
    path = tmp_path/"source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("12.0.swc", "1 3 0 0 0 10 -1\n2 0 100 0 0 10 1\n")
        archive.writestr("12.1.swc", "1 0 50 10 0 10 -1\n")
    archive = SimpleNamespace(path=path, components=lambda _: (0, 1))
    result = module._nearest_component(archive, "12", np.array([1.6, 0., 0.]))
    assert result["component"] == 0 and result["distance_um"] == 0.
    assert result["has_soma"] and not result["ambiguous"]
    with zipfile.ZipFile(path, "a") as output:
        output.writestr("12.2.swc", "1 0 0 0 0 10 -1\n2 0 100 0 0 10 1\n")
    archive.components = lambda _: (0, 1, 2)
    assert module._nearest_component(archive, "12", np.array([1.6, 0., 0.]))["ambiguous"]


@pytest.mark.parametrize("text", ["1 0 0 0 0 10 -1\n", "1 3 0 0 0 10 -1\n2 0 0 0 0 10 1\n"])
def test_missing_or_zero_length_cable_rejected(tmp_path, text):
    path = tmp_path/"source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("12.0.swc", text)
    with pytest.raises(ValueError):
        module._nearest_component(SimpleNamespace(path=path, components=lambda _: (0,)), "12", np.zeros(3))


@pytest.mark.parametrize("distance", [0., -1., np.nan])
def test_invalid_projection_limit(distance):
    with pytest.raises(ValueError, match="distance"):
        module.prepare_connectivity("unused", "unused", max_distance_um=distance)


@pytest.mark.parametrize("case", ["ready", "fragment", "distant", "ambiguous", "projection", "inventory"])
def test_prepare_retains_construction_blockers(imported, annotations, tmp_path, monkeypatch, case):
    path = tmp_path/"audit.json"
    path.write_text(json.dumps(report()))
    archive = SimpleNamespace(neuron_ids=("12", "13", "14"),
        load=lambda identity, component: replace(imported, neuron_id=identity))
    monkeypatch.setattr(module, "H01Archive", lambda _: archive)
    monkeypatch.setattr(module, "H01Annotations", lambda _: annotations)
    placement = dict(component=0, source_swc_sha256=imported.source_sha256,
                     has_soma=case != "fragment", ambiguous=case == "ambiguous",
                     distance_um=2. if case == "distant" else 0.)
    monkeypatch.setattr(module, "_nearest_component", lambda *a: deepcopy(placement))
    if case == "inventory":
        archive.neuron_ids = ("12",)
        with pytest.raises(ValueError, match="inventories"):
            module.prepare_connectivity(tmp_path, path)
        return
    if case == "projection":
        archive.load = lambda *a, **kw: SimpleNamespace(anatomy=lambda: SimpleNamespace(
            project=lambda *a, **kw: [SimpleNamespace(status="ambiguous")]))
    result = module.prepare_connectivity(tmp_path, path)
    assert result["counts"]["construction_ready"] == int(case == "ready")
    assert bool(result["contacts"][0]["blockers"]) == (case != "ready")


def c3_row(pre=1001, post=1002, kind=2, index=0, **extra):
    row = dict(pre=pre, post=post, type=kind, x=800, y=0, z=0, pre_class="pyramidal", post_class="interneuron",
               confidence=.9, shard=3, index=index)
    row.update(extra)
    return row


C3_CELLS = [dict(cell_id="12", tags=["pyramidal", "L2"], c3_id="1001"),
            dict(cell_id="13", tags=["interneuron"], c3_id="1002", archive_sha256="b"*64),
            dict(cell_id="14", polarity="E", c3_id="1003", archive_sha256="b"*64)]
C3_MAP = {"1001": "12", "1002": "13", 1003: "14"}


def test_select_c3_connectivity_maps_identities_and_signs():
    rows = [c3_row(), c3_row(1002, 1001, 1, 1), c3_row(1003, 1002, 2, shard=None, index=None)]
    result = module.select_c3_connectivity(rows, C3_CELLS, C3_MAP)
    assert result["schema"] == module.C3_SELECTION_SCHEMA and result["archives"] == [module.ARCHIVE_SHA256, "b"*64]
    assert [n["dale_sign"] for n in result["nodes"]] == [1, -1, 1] and result["nodes"][2]["polarity"] == "E"
    assert result["nodes"][0]["archive_sha256"] == module.ARCHIVE_SHA256 and result["nodes"][1]["c3_id"] == "1002"
    contacts = result["contacts"]
    assert [e["annotation_id"] for e in contacts][:2] == ["c3-3-0", "c3-3-1"] and contacts[2]["annotation_id"].startswith("c3-")
    assert len(contacts[2]["annotation_id"]) == 19 and result["excluded_contacts"] == []
    assert contacts[0]["pre_cell"] == "12" and contacts[0]["post_cell"] == "13" and contacts[0]["connection_type"] == "EI"
    assert contacts[1]["connection_type"] == "IE" and contacts[1]["dale_sign"] == -1
    assert contacts[0]["pre_voxel"] == [800, 0, 0] == contacts[0]["post_voxel"]
    assert contacts[0]["endpoints_verified"] is True and contacts[0]["identity_basis"] == "c3-neuron-id"
    assert contacts[0]["pre_proofread_label"] == "12" and contacts[0]["pre_nearest_offset"] == [0, 0, 0]
    assert contacts[0]["c3_pre"] == "1001" and contacts[0]["confidence"] == .9
    assert rows[0]["x"] == 800   # input untouched


@pytest.mark.parametrize("row, reason", [
    (c3_row(1001, 9999), "endpoint cell absent from node inventory"),
    (c3_row(1001, 1001), "presynaptic and postsynaptic cell are the same"),
    (c3_row(1001, 1002, 1), "synapse type conflicts with presynaptic Dale sign"),
    (c3_row(1002, 1001, 2), "synapse type conflicts with presynaptic Dale sign"),
    (c3_row(x=-1), "invalid endpoint coordinate"),
    (c3_row(z=np.nan), "invalid endpoint coordinate"),
])
def test_select_c3_connectivity_excludes_with_reasons(row, reason):
    result = module.select_c3_connectivity([row], C3_CELLS, C3_MAP)
    assert result["contacts"] == [] and result["excluded_contacts"] == [dict(annotation_id="c3-3-0", reason=reason)]


def test_select_c3_connectivity_duplicates_and_bad_cells():
    result = module.select_c3_connectivity([c3_row(), c3_row()], C3_CELLS, C3_MAP)
    assert len(result["contacts"]) == 1 and result["excluded_contacts"][0]["reason"] == "duplicate annotation"
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        module.select_c3_connectivity([c3_row(), c3_row(confidence=.1)], C3_CELLS, C3_MAP)
    with pytest.raises(ValueError, match="Duplicate cell"):
        module.select_c3_connectivity([], C3_CELLS+[dict(cell_id="12", tags=["pyramidal"])], C3_MAP)
    with pytest.raises(ValueError, match="unambiguous"):
        module.select_c3_connectivity([], [dict(cell_id="1", tags=["neuron"])], {})


def test_c3_annotation_id_hash_is_stable_and_order_free():
    row = dict(pre=1, post=2, type=2, x=1, y=2, z=3)
    assert module.c3_annotation_id(row) == module.c3_annotation_id(dict(reversed(list(row.items()))))
    assert module.c3_annotation_id(row) != module.c3_annotation_id(dict(row, x=2))
    assert module.c3_annotation_id(dict(row, shard=7, index=12)) == "c3-7-12"


def test_archive_digest_and_archives_by_digest(tmp_path):
    path = tmp_path/"a.zip"
    path.write_bytes(b"archive")
    unpinned = SimpleNamespace(path=path)
    assert module.archive_digest(unpinned) == module._digest(path)
    assert module.archive_digest(SimpleNamespace(path=path, expected_sha256="c"*64)) == "c"*64
    listed = module.archives_by_digest(SimpleNamespace(archives=lambda: [unpinned]))
    assert list(listed) == [module._digest(path)]
    mapped = module.archives_by_digest(SimpleNamespace(archives=lambda: {"d"*64: unpinned}))
    assert mapped == {"d"*64: unpinned}


def _c3_selection(tmp_path):
    path = tmp_path/"c3-selection.json"
    selection = module.select_c3_connectivity([c3_row()], C3_CELLS, C3_MAP)
    path.write_text(json.dumps(selection))
    return path


class _Set:
    def __init__(self, by_digest, load):
        self.by_digest, self._load, self.calls = by_digest, load, []

    def archives(self):
        return dict(self.by_digest)

    def load(self, cell, component, digest):
        self.calls.append((cell, component, digest))
        return self._load(cell, component)


@pytest.mark.parametrize("distance, blocker, ready, within", [
    (0., None, True, True), (2., None, False, False), (2., 3., True, False), (0., 3., True, True), (3.5, 3., False, False)])
def test_prepare_c3_selection_records_distance_and_blocks_only_beyond_the_blocker(
        imported, tmp_path, monkeypatch, distance, blocker, ready, within):
    path = _c3_selection(tmp_path)
    load = lambda identity, component: replace(imported, neuron_id=identity)
    archives = _Set({module.ARCHIVE_SHA256: SimpleNamespace(neuron_ids=("12",)),
                     "b"*64: SimpleNamespace(neuron_ids=("13", "14"))}, load)
    limits, real_project = [], type(imported.anatomy()).project
    def project(anatomy, points, max_distance_um):
        limits.append(max_distance_um)
        return real_project(anatomy, points, max_distance_um=max_distance_um)
    monkeypatch.setattr(module, "_nearest_component", lambda *a: dict(component=0, source_swc_sha256=imported.source_sha256,
                                                                      has_soma=True, ambiguous=False, distance_um=distance))
    monkeypatch.setattr(type(imported.anatomy()), "project", project)
    result = module.prepare_connectivity(tmp_path, path, archive_set=archives, blocker_distance_um=blocker)
    edge = result["contacts"][0]
    assert edge["construction_ready"] is ready and edge["pre_placement"]["placement_distance_um"] == distance
    assert edge["pre_placement"]["within_max_distance"] is within
    assert result["max_distance_um"] == 1. and result["blocker_distance_um"] == blocker
    assert result["counts"]["beyond_max_distance"] == (0 if within else 2)
    if ready:
        assert archives.calls == [("12", 0, module.ARCHIVE_SHA256), ("13", 0, "b"*64)]
        assert set(limits) == {1. if blocker is None else blocker}
        assert "cable_location" in edge["post_placement"]
    else:
        assert archives.calls == [] and edge["blockers"][0].startswith("pre:")


def test_prepare_c3_selection_needs_a_set_holding_every_node(tmp_path, monkeypatch):
    path = _c3_selection(tmp_path)
    with pytest.raises(ValueError, match="archive set"):
        module.prepare_connectivity(tmp_path, path)
    with pytest.raises(ValueError, match="does not hold"):
        module.prepare_connectivity(tmp_path, path, archive_set=_Set({module.ARCHIVE_SHA256: SimpleNamespace(neuron_ids=("12",))}, None))
    absent = _Set({module.ARCHIVE_SHA256: SimpleNamespace(neuron_ids=("12",)), "b"*64: SimpleNamespace(neuron_ids=("13",))}, None)
    with pytest.raises(ValueError, match="absent from its archive"):
        module.prepare_connectivity(tmp_path, path, archive_set=absent)
    with pytest.raises(ValueError, match="Blocker distance"):
        module.prepare_connectivity(tmp_path, path, archive_set=absent, blocker_distance_um=.5)


def test_prepare_proofread_audit_keeps_the_single_archive_route(imported, annotations, tmp_path, monkeypatch):
    path = tmp_path/"audit.json"
    path.write_text(json.dumps(report()))
    archive = SimpleNamespace(neuron_ids=("12", "13", "14"), load=lambda identity, component: replace(imported, neuron_id=identity))
    monkeypatch.setattr(module, "H01Archive", lambda _: archive)
    monkeypatch.setattr(module, "H01Annotations", lambda _: annotations)
    monkeypatch.setattr(module, "_nearest_component", lambda *a: dict(component=0, source_swc_sha256=imported.source_sha256,
                                                                      has_soma=True, ambiguous=False, distance_um=2.5))
    result = module.prepare_connectivity(tmp_path, path, blocker_distance_um=3.)
    assert result["contacts"][0]["construction_ready"] is True and result["blocker_distance_um"] == 3.
    assert result["contacts"][0]["pre_placement"]["within_max_distance"] is False
