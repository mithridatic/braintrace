"""Check report mathematics without requiring NEURON on the Windows host."""

import ast
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("times", [np.array([0., 200., 201., 202., 269., 270., 1500.]),
                                    np.arange(0., 1501.)])
def test_baseline_mean_is_time_weighted_for_adaptive_samples(times):
    tree = ast.parse(Path(__file__).with_name("h01_pv_neuron_reference.py").read_text())
    scope = {"np": np, "times": times, "voltage": times.copy()}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "reference_helpers", "exec"), scope)
    report = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "report" for t in node.targets))
    expression = next(v for k, v in zip(report.keys, report.values) if k.value == "baseline_mean_mv")
    actual = eval(compile(ast.Expression(expression), "baseline_report", "eval"), scope)
    # The time average of V(t)=t from 200 to 270 is 235, independent of samples.
    assert actual == pytest.approx(235.)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "5", "0.002", None])
@pytest.mark.parametrize("option", ["--sodium-h-slope-mv", "--somatic-calva-factor", "--somatic-kv3-factor", "--somatic-kv3-tau-factor", "--somatic-kv3-close-factor", "--axon-calcium-gamma"])
def test_channel_parameter_cli_boundary(monkeypatch, value, option):
    """Reject invalid parameters before construction and preserve valid defaults."""
    tree = ast.parse(Path(__file__).with_name("h01_pv_neuron_reference.py").read_text())
    nodes = []
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "h"):
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    argv = ["reference", "--current-na", ".19", "--output", "unused"]
    if value is not None:
        argv += [option, value]
    monkeypatch.setattr("sys.argv", argv)
    scope = {"np": np, "argparse": argparse, "Path": Path, "hashlib": hashlib, "json": json, "__doc__": "CLI validation"}
    code = compile(ast.Module(body=nodes, type_ignores=[]), "reference_cli", "exec")
    invalid = (value in ("-1", "nan", "inf")
               or (value == "0" and option in ("--sodium-h-slope-mv", "--somatic-kv3-tau-factor", "--somatic-kv3-close-factor"))
               or (value == "5" and option == "--axon-calcium-gamma"))
    if invalid:
        with pytest.raises(SystemExit) as error:
            exec(code, scope)
        assert error.value.code == 2
    else:
        exec(code, scope)
        default = None if option in ("--sodium-h-slope-mv", "--somatic-kv3-close-factor", "--axon-calcium-gamma") else 1.
        attribute = option.removeprefix("--").replace("-", "_")
        assert getattr(scope["args"], attribute) == (default if value is None else float(value))


from types import SimpleNamespace

SOURCE = Path(__file__).with_name("h01_pv_neuron_reference.py")


@pytest.mark.parametrize("region,expected", [
    ("all", [81, 135, 189, 297]),
    ("soma", [81, 45, 63, 99]),
    ("axon", [27, 135, 63, 99]),
    ("dendrites", [27, 45, 189, 297]),
])
def test_regional_allocation_preserves_unselected_baseline(region, expected):
    """Execute the driver's real allocation on unequal source segment counts."""
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.For)
                and "args.refine_region" in ast.unparse(n))
    sections = [SimpleNamespace(nseg=count, name=lambda f=family: f"Cell[0].{f}[0]")
                for family, count in zip(("soma", "axon", "dend", "apic"), (3, 5, 7, 11))]
    scope = {"cell": SimpleNamespace(all=sections),
             "args": SimpleNamespace(refine_region=region, nseg_factor=27, unselected_nseg_factor=9)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "mesh_allocation", "exec"), scope)
    assert [s.nseg for s in sections] == expected


@pytest.mark.parametrize("value", [None, "1", "9", "0", "-1", "2", "2.5"])
def test_unselected_factor_cli_boundary(monkeypatch, value):
    """Validate defaults and reject invalid counts before model construction."""
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("h.load_file"):
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    argv = ["reference", "--current-na", ".27", "--output", "unused"]
    if value is not None:
        argv += ["--unselected-nseg-factor", value]
    monkeypatch.setattr("sys.argv", argv)
    scope = {"np": np, "argparse": argparse, "Path": Path, "hashlib": hashlib, "json": json, "__doc__": "CLI test"}
    code = compile(ast.Module(body=nodes, type_ignores=[]), "mesh_cli", "exec")
    if value in ("0", "-1", "2", "2.5"):
        with pytest.raises(SystemExit) as error:
            exec(code, scope)
        assert error.value.code == 2
    else:
        exec(code, scope)
        assert scope["args"].unselected_nseg_factor == (1 if value is None else int(value))


@pytest.mark.parametrize("value", [None, "330", "0", "nan", "-5"])
def test_duration_cli_boundary(monkeypatch, value):
    """Validate the default and reject a non-positive or non-finite duration."""
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("h.load_file"):
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    argv = ["reference", "--current-na", ".27", "--output", "unused"]
    if value is not None:
        argv += ["--duration-ms", value]
    monkeypatch.setattr("sys.argv", argv)
    scope = {"np": np, "argparse": argparse, "Path": Path, "hashlib": hashlib, "json": json, "__doc__": "CLI test"}
    code = compile(ast.Module(body=nodes, type_ignores=[]), "duration_cli", "exec")
    if value in ("0", "nan", "-5"):
        with pytest.raises(SystemExit) as error:
            exec(code, scope)
        assert error.value.code == 2
    else:
        exec(code, scope)
        assert scope["args"].duration_ms == (1500. if value is None else float(value))


def test_simulation_and_bias_follow_the_requested_duration():
    """No literal 1500 ms remains in the run, bias clamp, or report."""
    text = SOURCE.read_text()
    assert "h.continuerun(args.duration_ms)" in text
    assert "bias.dur = args.duration_ms+1." in text
    assert '"duration_ms": args.duration_ms' in text
    assert "continuerun(1500" not in text and '"duration_ms": 1500' not in text
    assert '"mechanism_library"' in text


def _cli_scope(monkeypatch, argv):
    """Execute the driver's CLI section (everything before the first h.load_file) with ``argv``."""
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("h.load_file"):
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    monkeypatch.setattr("sys.argv", ["reference", "--current-na", ".27", "--output", "unused", *argv])
    scope = {"np": np, "argparse": argparse, "Path": Path, "hashlib": hashlib, "json": json, "__doc__": "scale CLI test"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "scale_cli", "exec"), scope)
    return scope


def test_scale_is_repeatable_and_parsed_like_the_l2_regional_density(monkeypatch):
    scope = _cli_scope(monkeypatch, ["--scale", "NaTg:axon:1.5", "--scale", "SK:all:0"])
    assert scope["regional_scales"] == [{"mechanism": "NaTg", "region": "axon", "factor": 1.5},
                                        {"mechanism": "SK", "region": "all", "factor": 0.}]
    assert scope["args"].scale == ["NaTg:axon:1.5", "SK:all:0"]


def test_scale_defaults_leave_the_single_slot_untouched(monkeypatch):
    scope = _cli_scope(monkeypatch, ["--scale-conductance", "NaTg", "--conductance-factor", "1.1", "--conductance-region", "soma"])
    assert scope["regional_scales"] == [] and scope["args"].scale == []
    args = scope["args"]
    assert (args.scale_conductance, args.conductance_factor, args.conductance_region) == ("NaTg", 1.1, "soma")


@pytest.mark.parametrize("text", ["NaTg:axon", "Nap:axon:1", "NaTg:apex:1", "NaTg:soma:nan", "NaTg:all:-1", "NaTg:soma:inf"])
def test_scale_rejects_malformed_text_before_construction(monkeypatch, text):
    with pytest.raises(SystemExit) as error:
        _cli_scope(monkeypatch, ["--scale", text])
    assert error.value.code == 2


def test_candidate_json_scale_list_is_a_known_flag_and_cli_appends(monkeypatch, tmp_path):
    candidate = tmp_path/"arm.candidate.json"
    candidate.write_text(json.dumps({"scale": ["NaTg:axon:1.5"], "scale_conductance": "NaTg", "conductance_factor": 1.1,
                                     "conductance_region": "soma"}))
    scope = _cli_scope(monkeypatch, ["--candidate-json", str(candidate), "--scale", "Kv3_1:soma:2"])
    assert [s["mechanism"] for s in scope["regional_scales"]] == ["NaTg", "Kv3_1"]
    assert scope["args"].conductance_factor == 1.1
    assert scope["candidate_record"]["values"]["scale"] == ["NaTg:axon:1.5"]


def test_regional_scale_applies_after_the_single_slot_and_raises_on_no_match():
    """Execute the driver's real scaling loops on stand-in sections."""
    tree = ast.parse(SOURCE.read_text())
    loops = [n for n in tree.body if isinstance(n, (ast.If, ast.For))
             and ("args.scale_conductance is not None" in ast.unparse(n).splitlines()[0]
                  or ast.unparse(n).startswith("for scale in regional_scales"))]
    assert len(loops) == 2

    class Section:
        def __init__(self, family, mechs, gbar):
            self._name, self._mechs, self.gbar_NaTg = f"Cell[0].{family}[0]", mechs, gbar

        def name(self):
            return self._name

        def psection(self):
            return {"density_mechs": self._mechs}

    sections = [Section("soma", {"NaTg": {}}, 2.), Section("axon", {"NaTg": {}}, 4.), Section("dend", {}, 8.)]
    scope = {"cell": SimpleNamespace(all=sections),
             "args": SimpleNamespace(scale_conductance="NaTg", conductance_factor=1.1, conductance_region="soma"),
             "regional_scales": [{"mechanism": "NaTg", "region": "axon", "factor": 1.5}]}
    exec(compile(ast.Module(body=loops, type_ignores=[]), "scaling", "exec"), scope)
    assert [s.gbar_NaTg for s in sections] == pytest.approx([2.2, 6., 8.])
    scope["regional_scales"] = [{"mechanism": "NaTg", "region": "dend", "factor": 1.5}]
    with pytest.raises(RuntimeError, match="matched no section"):
        exec(compile(ast.Module(body=loops, type_ignores=[]), "scaling", "exec"), scope)


def test_mechanism_parameter_is_repeatable_and_rejects_malformed_text(monkeypatch):
    scope = _cli_scope(monkeypatch, ["--mechanism-parameter", "NaTg:all:slow_inactivation:0.3",
                                     "--mechanism-parameter", "NaTg:soma:s_tau_entry_ms:10"])
    assert scope["mechanism_parameters"] == [
        {"mechanism": "NaTg", "region": "all", "name": "slow_inactivation", "value": .3},
        {"mechanism": "NaTg", "region": "soma", "name": "s_tau_entry_ms", "value": 10.}]
    for text in ("NaTg:all:slow_inactivation", "NaTg:apex:slow_inactivation:0.3", "NaTg:all:slow_inactivation:nan"):
        with pytest.raises(SystemExit) as error:
            _cli_scope(monkeypatch, ["--mechanism-parameter", text])
        assert error.value.code == 2


def test_mechanism_parameter_sets_every_matching_section_and_raises_on_no_match_or_no_parameter():
    tree = ast.parse(SOURCE.read_text())
    loops = [n for n in tree.body if isinstance(n, ast.For) and ast.unparse(n).startswith("for item in mechanism_parameters")]
    assert len(loops) == 1

    class Section:
        def __init__(self, family, mechs):
            self._name, self._mechs, self.slow_inactivation_NaTg = f"Cell[0].{family}[0]", mechs, 0.

        def name(self):
            return self._name

        def psection(self):
            return {"density_mechs": self._mechs}

    sections = [Section("soma", {"NaTg": {}}), Section("axon", {"NaTg": {}}), Section("dend", {})]
    scope = {"cell": SimpleNamespace(all=sections),
             "mechanism_parameters": [{"mechanism": "NaTg", "region": "all", "name": "slow_inactivation", "value": .3}]}
    exec(compile(ast.Module(body=loops, type_ignores=[]), "parameters", "exec"), scope)
    assert [s.slow_inactivation_NaTg for s in sections] == [.3, .3, 0.]
    scope["mechanism_parameters"] = [{"mechanism": "NaTg", "region": "dend", "name": "slow_inactivation", "value": .3}]
    with pytest.raises(RuntimeError, match="matched no section"):
        exec(compile(ast.Module(body=loops, type_ignores=[]), "parameters", "exec"), scope)
    scope["mechanism_parameters"] = [{"mechanism": "NaTg", "region": "soma", "name": "s_vhalf", "value": -50.}]
    with pytest.raises(RuntimeError, match="no RANGE parameter"):
        exec(compile(ast.Module(body=loops, type_ignores=[]), "parameters", "exec"), scope)


def test_report_records_the_regional_scales():
    text = SOURCE.read_text()
    assert '"regional_scales": regional_scales' in text
    assert '"mechanism_parameters": mechanism_parameters' in text
    assert '"conductance_intervention": {"mechanism": args.scale_conductance' in text


def _cli_scope(monkeypatch, argv):
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("h.load_file"):
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    monkeypatch.setattr("sys.argv", ["reference", "--current-na", ".1", "--output", "unused", *argv])
    scope = {"np": np, "argparse": argparse, "Path": Path, "hashlib": hashlib, "json": json, "__doc__": "donor CLI"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "donor_cli", "exec"), scope)
    return scope


def test_donor_flags_default_to_the_hl5bn1_files(monkeypatch):
    """Today's behaviour is the default, so every retained candidate report is unchanged."""
    scope = _cli_scope(monkeypatch, [])
    assert (scope["args"].template, scope["args"].biophys, scope["args"].morphology) == (
        "/work/source/NeuronTemplate.hoc", "/work/source/biophys_HL5BN1.hoc", "/work/source/HL5BN1.swc")
    assert scope["biophys_procedure"] == "biophys_HL5BN1"


def test_donor_flags_select_hl5mn1_and_derive_its_procedure(monkeypatch):
    scope = _cli_scope(monkeypatch, ["--biophys", "/work/source/biophys_HL5MN1.hoc",
                                     "--morphology", "/work/morphologies/HL5MN1.swc"])
    assert scope["args"].morphology == "/work/morphologies/HL5MN1.swc"
    assert scope["biophys_procedure"] == "biophys_HL5MN1"
    assert scope["_biophys_procedure"]("C:/x/biophys_HL23SST.hoc") == "biophys_HL23SST"


@pytest.mark.parametrize("name", ["HL5MN1.hoc", "biophys_HL5MN1.txt", "/work/source/template.hoc"])
def test_misnamed_biophys_file_is_rejected_before_construction(monkeypatch, name):
    with pytest.raises(SystemExit) as error:
        _cli_scope(monkeypatch, ["--biophys", name])
    assert error.value.code == 2


def test_candidate_json_accepts_the_donor_flags(monkeypatch, tmp_path):
    candidate = tmp_path/"source.candidate.json"
    candidate.write_text(json.dumps({"biophys": "/work/source/biophys_HL5MN1.hoc",
                                     "morphology": "/work/morphologies/HL5MN1.swc", "initial_mv": -81.5, "dt_ms": .025}))
    scope = _cli_scope(monkeypatch, ["--candidate-json", str(candidate)])
    assert scope["biophys_procedure"] == "biophys_HL5MN1" and scope["args"].initial_mv == -81.5
    assert scope["candidate_record"]["file"] == "source.candidate.json"


class _DeletedSectionArray:
    """Stand-in for a hoc section array whose only section was deleted: any access aborts."""

    def __getitem__(self, index):
        raise RuntimeError("section in the object was deleted")

    def __iter__(self):
        raise RuntimeError("section in the object was deleted")


def test_myelin_geometry_skips_a_template_array_of_deleted_sections(monkeypatch):
    """HL5BN1's template deletes ``myelin[1]`` and never recreates it; the driver must not touch it."""
    scope = _cli_scope(monkeypatch, [])
    live = {("myelin", 0): False}
    scope["h"] = SimpleNamespace(section_exists=lambda name, index, cell: float(live.get((name, index), False)))
    cell = SimpleNamespace(myelin=_DeletedSectionArray())
    assert scope["_existing_sections"](cell, "myelin") == []
    section = object()
    live[("myelin", 0)] = True
    cell = SimpleNamespace(myelin={0: section})
    assert scope["_existing_sections"](cell, "myelin") == [section]
    text = SOURCE.read_text()
    assert "for m in cell.myelin" not in text
    assert '_existing_sections(cell, "myelin")' in text
