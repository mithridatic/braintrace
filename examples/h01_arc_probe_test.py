"""Failure evidence and resource-limit validation for the diagnostic runner."""

import json
import sys

import pytest

from examples import h01_arc_probe


def test_failed_construction_keeps_phase_evidence(tmp_path, monkeypatch):
    output = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--cells", "4", "--cache", str(tmp_path),
                                     "--output", str(output)])
    monkeypatch.setattr(h01_arc_probe, "source_archives", lambda paths, topology: (object(), {}))
    monkeypatch.setattr(h01_arc_probe, "H01Annotations", lambda _: object())
    def fail(*args, **kwargs):
        raise ValueError("invalid pinned population")
    monkeypatch.setattr(h01_arc_probe, "make_h01_network", fail)
    h01_arc_probe.main()
    result = json.loads(output.read_text())
    assert result["status"] == "blocked"
    assert result["error"] == "invalid pinned population"
    assert result["phases"]["construction"]["status"] == "failed"
    assert result["phases"]["construction"]["seconds"] >= 0
    assert result["limits"] == {"wall_seconds": 900., "rss_gib": 16.}


@pytest.mark.parametrize("flag,value", [("--rss-limit-gib", "0"),
    ("--rss-limit-gib", "nan"), ("--wall-limit-seconds", "-1")])
def test_invalid_limits_rejected_before_construction(tmp_path, monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["probe", "--cells", "4", "--cache", str(tmp_path),
        "--output", str(tmp_path/"unused.json"), flag, value])
    with pytest.raises(SystemExit) as exc:
        h01_arc_probe.main()
    assert exc.value.code == 2
    assert not (tmp_path/"unused.json").exists()


def test_sparse_probe_uses_grouped_muon_on_actual_cable(tmp_path):
    from braintrace.datasets.h01_network_step_test import _contact_network
    from examples.pp_prop.h01_arc_model import H01ArcModel
    model = H01ArcModel(_contact_network(tmp_path), ['5805562981', '5965472721'])
    report = {}
    phases = []
    def phase(name, call):
        value = call()
        phases.append(name)
        return value
    h01_arc_probe._sparse_learning_probe(model, report, phase)
    assert phases == ['sparse_pp_prop_compile', 'sparse_muon_compile_and_update', 'sparse_muon_warm_update']
    groups = report['optimizer']['groups']
    assert groups['input']['algorithm'] == groups['recurrent']['algorithm'] == 'masked_muon'
    assert groups['readout_weight']['algorithm'] == 'muon'
    assert groups['readout_bias']['algorithm'] == 'adamw'
    assert all(group['weight_decay'] == .1 for group in groups.values())
    assert report['learning_probe']['finite']
    assert report['learning_probe']['updates'] == 2
    assert report['learning_probe']['cold_gradient_norm'] > 0
    assert set(report['learning_probe']['changed_parameters']) == {
        'input', 'recurrent', 'readout_weight', 'readout_bias'}


def _network():
    return dict(nodes=[dict(cell_id="1"), dict(cell_id="2", archive_sha256="b"*64), dict(cell_id="3")],
                contacts=[dict(pre_cell="1", post_cell="2"), dict(pre_cell="2", post_cell="3")], max_distance_um=1.)


def test_restrict_topology_keeps_listed_nodes_and_contacts_among_them():
    result = h01_arc_probe.restrict_topology(_network(), ["2", "1"])
    assert [n["cell_id"] for n in result["nodes"]] == ["1", "2"]
    assert result["contacts"] == [dict(pre_cell="1", post_cell="2")] and result["max_distance_um"] == 1.
    with pytest.raises(ValueError, match="absent from the network nodes: 9"):
        h01_arc_probe.restrict_topology(_network(), ["1", "9"])
    with pytest.raises(ValueError, match="at least one"):
        h01_arc_probe.restrict_topology(_network(), [])


def test_source_archives_routes_each_node_through_its_digest(monkeypatch, tmp_path):
    from types import SimpleNamespace
    calls = []
    archive_set = SimpleNamespace(archives=lambda: [], load=lambda identity, component, digest:
                                  calls.append((identity, component, digest)) or "component")
    monkeypatch.setattr(h01_arc_probe, "open_archive_paths",
                        lambda paths: (archive_set, {str(p): d for p, d in zip(paths, ("a"*64, "b"*64))}))
    monkeypatch.setattr(h01_arc_probe, "ARCHIVE_SHA256", "a"*64)
    view, digests = h01_arc_probe.source_archives([tmp_path/"x.zip", tmp_path/"y.zip"], _network())
    assert view.neuron_ids == ("1", "2", "3") and list(digests.values()) == ["a"*64, "b"*64]
    assert view.load("2", component=4) == "component" and view.load(1, component=0) == "component"
    assert calls == [("2", 4, "b"*64), ("1", 0, "a"*64)]
    single = SimpleNamespace(load=lambda identity, component: "single")
    monkeypatch.setattr(h01_arc_probe, "open_archive_paths", lambda paths: (single, {"x": "a"*64}))
    assert h01_arc_probe.source_archives([tmp_path/"x.zip"], _network())[0] is single
    monkeypatch.setattr(h01_arc_probe, "open_archive_paths", lambda paths: (archive_set, {"x": "c"*64, "y": "d"*64}))
    with pytest.raises(ValueError, match="not opened"):
        h01_arc_probe.source_archives([tmp_path/"x.zip", tmp_path/"y.zip"], _network())


def test_overlaid_annotations_prefer_the_overlay():
    from types import SimpleNamespace
    base = SimpleNamespace(metadata=lambda identity: SimpleNamespace(tags=("pyramidal", "L2")))
    annotations = h01_arc_probe.OverlaidAnnotations(base, {900: ["interneuron", "L3"]})
    assert annotations.metadata("900").tags == ("interneuron", "L3")
    assert annotations.metadata("1").tags == ("pyramidal", "L2")


def test_cell_id_list_restricts_the_network_before_construction(tmp_path, monkeypatch):
    output, ids = tmp_path/"probe.json", tmp_path/"ids.json"
    ids.write_text(json.dumps(["2", "1"]))
    (tmp_path/"network.json").write_text(json.dumps(_network()))
    (tmp_path/"components.json").write_text("{}")
    (tmp_path/"tags.json").write_text(json.dumps({"2": ["interneuron"]}))
    monkeypatch.setattr(sys, "argv", ["probe", "--cell-ids", str(ids), "--cache", str(tmp_path),
                                     "--archive", str(tmp_path/"a.zip"), "--archive", str(tmp_path/"b.zip"),
                                     "--topology", str(tmp_path/"network.json"),
                                     "--components", str(tmp_path/"components.json"),
                                     "--annotations", str(tmp_path/"tags.json"), "--output", str(output)])
    seen = {}
    monkeypatch.setattr(h01_arc_probe, "source_archives", lambda paths, topology:
                        seen.update(paths=[p.name for p in paths], topology=topology) or ("archive", {"a": "x"}))
    monkeypatch.setattr(h01_arc_probe, "H01Annotations", lambda _: object())
    def fail(topology, archive, annotations, **kwargs):
        seen.update(archive=archive, tags=annotations.metadata("2").tags, cells=kwargs["cells"])
        raise ValueError("stop here")
    monkeypatch.setattr(h01_arc_probe, "make_h01_network", fail)
    h01_arc_probe.main()
    result = json.loads(output.read_text())
    assert result["cells"] == 2 and result["cell_ids"] == ["2", "1"] and result["archive_sha256"] == {"a": "x"}
    assert result["archives"] == [str(tmp_path/"a.zip"), str(tmp_path/"b.zip")]
    assert seen["paths"] == ["a.zip", "b.zip"] and seen["cells"] is None and seen["archive"] == "archive"
    assert [n["cell_id"] for n in seen["topology"]["nodes"]] == ["1", "2"] and seen["tags"] == ("interneuron",)
