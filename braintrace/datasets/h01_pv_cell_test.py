"""Published cell assembly and passive analytic checks."""

import json
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np
import pytest

from .h01_pv_cell import make_pv_cell

REFERENCE = json.loads((Path(__file__).resolve().parents[2] /
                       "docs/evidence/h01-pv-geometry-reference.json").read_text())


def test_passive_tree_follows_uniform_leak_relaxation():
    with brainstate.environ.context(precision=64):
        cell = make_pv_cell(REFERENCE, current_na=0., active=False)
        result = cell.run(dt=.01*u.ms, duration=20.*u.ms)
        voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
    time = .01*np.arange(1, len(voltage)+1)
    reversal = -96.97510324827309
    tau = 2./(.00022279797042703468*1000)
    expected = reversal+(-80.-reversal)*np.exp(-time/tau)
    np.testing.assert_allclose(voltage.ravel(), expected, atol=.004, rtol=0.)


def test_active_cell_initialization_and_calcium_are_finite():
    with brainstate.environ.context(precision=64):
        result = make_pv_cell(REFERENCE).run(dt=.005*u.ms, duration=2.*u.ms)
        voltage = np.asarray(result.traces["voltage"].to_decimal(u.mV))
        calcium = np.asarray(result.traces["calcium"].to_decimal(u.mM))
        gates = np.asarray(result.traces["sk_gate"])
    assert np.isfinite(voltage).all() and np.isfinite(calcium).all()
    assert np.all(calcium > 0)
    assert np.all((gates >= 0) & (gates <= 1))


@pytest.mark.parametrize("current,length", [(float("nan"), 10.), (.19, 0.), (.19, float("inf"))])
def test_invalid_cell_inputs_fail(current, length):
    with pytest.raises(ValueError, match="Finite current"):
        make_pv_cell(REFERENCE, current, max_cv_length_um=length)
