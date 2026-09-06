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
