"""Check report mathematics without requiring NEURON on the Windows host."""

import ast
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
    scope = {"np": np, "argparse": argparse, "__doc__": "CLI validation"}
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
    scope = {"np": np, "argparse": argparse, "__doc__": "CLI test"}
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
    scope = {"np": np, "argparse": argparse, "__doc__": "CLI test"}
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
