"""Donor geometry reference initialization and explicit source mode."""
import json
from pathlib import Path
import brainstate
import brainunit as u
import numpy as np
import pytest
from .h01_l2_cell import make_l2_cell
REFERENCE = json.loads((Path(__file__).resolve().parents[2]/"docs/evidence/h01-l2-geometry-reference.json").read_text())


@pytest.mark.parametrize("mode", ["source", "candidate"])
def test_donor_cell_runs_with_finite_calcium(mode):
    with brainstate.environ.context(precision=64):
        result = make_l2_cell(REFERENCE, mode=mode).run(dt=.001*u.ms, duration=.05*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert (np.asarray(result.traces["calcium"].to_decimal(u.mM)) > 0).all()


@pytest.mark.parametrize("kwargs", [{"current_na": np.nan}, {"duration_ms": 0.}, {"delay_ms": -1.}])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError, match="Finite current"):
        make_l2_cell(REFERENCE, **kwargs)
