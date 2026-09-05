"""Check report mathematics without requiring NEURON on the Windows host."""

import ast
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
