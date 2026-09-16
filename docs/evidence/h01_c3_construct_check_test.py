"""Tests for the C3 construct/initialise check (injected builder; no BrainCell build)."""

import json

import pytest

from docs.evidence import h01_c3_construct_check as module


def test_resolve_role_prefers_overrides_then_record_then_tags():
    assert module.resolve_role({}, "I", "x") == ("I", "x")
    assert module.resolve_role({"polarity": "I", "donor": "d"}) == ("I", "d")
    polarity, donor = module.resolve_role({"tags": ["L4", "pyramidal", "neuron"]})
    assert polarity == "E" and donor == "l4-pyramidal-allen-527952884"
    polarity, donor = module.resolve_role({"layer": "L3", "cell_class": "interneuron"})
    assert polarity == "I" and donor == module.donor_for_tags(["L3", "interneuron"], "I")
    assert module.resolve_role({}) == ("E", module.DEFAULT_DONOR_KEYS["E"])


def test_candidate_index_and_plan_take_the_largest_component_per_inventory_cell():
    candidates = module.candidate_index({"candidates": [{"c3_id": 5, "polarity": "I", "donor": "d"}, 7, {"x": 1}]})
    assert list(candidates) == ["5"]
    inventory = {"cells": [{"cell_id": "5", "largest_component": 2, "largest_has_soma": True, "soma_components": [2]},
                           {"cell_id": "6", "largest_component": 0, "largest_has_soma": False, "soma_components": [4]}]}
    jobs = module.plan(inventory, candidates)
    assert jobs[0] == {"cell_id": "5", "component": 2, "component_selection": "largest", "polarity": "I", "donor": "d"}
    assert jobs[1]["polarity"] == "E" and jobs[1]["component"] == 4 and jobs[1]["component_selection"] == "largest_soma_bearing"
    assert module.plan(inventory, candidates, "E", "z")[0]["donor"] == "z"


def test_check_records_failures_checkpoints_and_resumes(tmp_path):
    calls = []

    def builder(job):
        calls.append(job["cell_id"])
        if job["cell_id"] == "6":
            raise RuntimeError("no axon")
        return {"n_compartments": 3, "construction_seconds": 0.1, "init_seconds": 0.2}

    jobs = [{"cell_id": "5", "component": 0, "polarity": "E", "donor": "d"},
            {"cell_id": "6", "component": 1, "polarity": "I", "donor": "d"}]
    output = tmp_path / "check.json"
    result = module.check(jobs, builder, output, emit=lambda m: None)
    assert result["status"] == "completed" and result["passed"] is False and result["passed_count"] == 1
    assert result["cells"][1]["error"] == "RuntimeError: no axon" and result["cells"][0]["n_compartments"] == 3
    again = module.check(jobs + [{"cell_id": "7", "component": 0, "polarity": "E", "donor": "d"}], builder, output)
    assert calls == ["5", "6", "6", "7"] and again["cells_expected"] == 3 and again["passed_count"] == 2
    assert [c["cell_id"] for c in again["cells"]] == ["5", "6", "7"]
    assert json.loads(output.read_text())["passed"] is False


def test_main_with_injected_builder_writes_the_report(tmp_path):
    (tmp_path / "components.json").write_text(json.dumps({"cells": [{"cell_id": "9", "largest_component": 0}]}))
    (tmp_path / "cand.json").write_text(json.dumps([{"c3_id": "9", "tags": ["L2", "pyramidal", "neuron"]}]))
    result = module.main(["--archive", str(tmp_path / "a.zip"), "--components", str(tmp_path / "components.json"),
                          "--candidates", str(tmp_path / "cand.json"), "--expected-sha256", "ab",
                          "--source", "c3", "--output", str(tmp_path / "out.json")],
                         builder=lambda job: {"n_compartments": 1})
    assert result["passed"] is True and result["cells"][0]["donor"] == "l2-pyramidal-allen-541563728"
    assert result["source"] == "c3" and result["expected_sha256"] == "ab"
    assert json.loads((tmp_path / "out.json").read_text())["passed_count"] == 1
    with pytest.raises(SystemExit):
        module.main(["--components", str(tmp_path / "components.json"), "--output", str(tmp_path / "o.json")])


def test_load_annotations_uses_the_c3_table_when_given(tmp_path):
    table = {"@type": "neuroglancer_segment_properties", "inline": {"ids": ["9"], "properties": [
        {"id": "tags", "type": "tags", "tags": ["L2", "pyramidal"], "values": [[0, 1]]}]}}
    (tmp_path / "c3.json").write_text(json.dumps(table))
    annotations = module.load_annotations(tmp_path, tmp_path / "c3.json")
    assert annotations.metadata("9").tags == ("L2", "pyramidal")
    with pytest.raises(OSError):
        module.load_annotations(tmp_path)


def test_select_component_takes_the_largest_soma_bearing_component():
    assert module.select_component({"largest_component": 0, "largest_has_soma": True, "soma_components": [0]}) == (0, "largest")
    row = {"largest_component": 0, "largest_has_soma": False, "soma_components": [3, 7]}
    assert module.select_component(row) == (3, "largest_soma_bearing")
    row = {"largest_component": 1, "largest_has_soma": False, "soma_components": []}
    assert module.select_component(row) == (1, "no_soma_component")


def test_production_builder_wires_import_build_register_and_init(monkeypatch):
    """The closure's call sequence, with the BrainCell-side functions replaced by recorders."""
    import contextlib
    import sys
    import types

    calls = []
    fake_braincell = types.SimpleNamespace(Network=lambda name: calls.append(("network", name)) or object())
    fake_brainstate = types.SimpleNamespace(environ=types.SimpleNamespace(context=lambda **kw: contextlib.nullcontext()))
    ei_cell = types.ModuleType("braintrace.datasets.h01_ei_cell")
    ei_cell.make_h01_ei_cell = lambda imported, annotations, **kw: (
        calls.append(("make", kw["polarity"], kw["donor"], kw["max_cv_length_um"], kw["solver"])) or ("cell", {"n_compartments": 12}))
    network = types.ModuleType("braintrace.datasets.h01_network")
    network._regions = lambda imported: calls.append(("regions",)) or {}
    network._register_cell = lambda net, identity, cell, evidence, imported, sites, current, emit: calls.append(("register", identity))
    init = types.ModuleType("braintrace.datasets.h01_network_init")
    init.init_h01_network_states = lambda net, progress, heartbeat_seconds: calls.append(("init",))
    init.process_rss_mb = lambda: 123.0
    for name, mod in [("braincell", fake_braincell), ("brainstate", fake_brainstate), (ei_cell.__name__, ei_cell),
                      (network.__name__, network), (init.__name__, init)]:
        monkeypatch.setitem(sys.modules, name, mod)
    imported = types.SimpleNamespace(source_rows=[1, 2, 3], source_sha256="abc")
    archive = types.SimpleNamespace(load=lambda cell_id, component: imported)
    build = module.production_builder(archive, object(), 10., "solver-x", 0.1, lambda m: None)
    record = build({"cell_id": "9", "component": 2, "polarity": "E", "donor": "d"})
    assert record["component_nodes"] == 3 and record["source_sha256"] == "abc"
    assert record["n_compartments"] == 12 and record["rss_mb"] == 123.0
    assert record["construction_seconds"] >= 0 and record["init_seconds"] >= 0
    assert [c[0] for c in calls] == ["regions", "make", "network", "register", "init"]
    assert calls[1] == ("make", "E", "d", 10., "solver-x") and calls[3] == ("register", "9")
