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
    inventory = {"cells": [{"cell_id": "5", "largest_component": 2}, {"cell_id": "6", "largest_component": 0}]}
    jobs = module.plan(inventory, candidates)
    assert jobs[0] == {"cell_id": "5", "component": 2, "polarity": "I", "donor": "d"}
    assert jobs[1]["polarity"] == "E" and jobs[1]["component"] == 0
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
