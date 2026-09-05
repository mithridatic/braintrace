"""Reference-compartment numerical qualification, separate from biology."""

import brainstate
import brainunit as u
import numpy as np
import pytest

from .h01_reference import make_reference_cell
from ._wilbers_oracle_test import oracle_trace


@pytest.mark.parametrize("kwargs", [{"current_na": np.nan}, {"sodium_ms_cm2": -1},
                                   {"potassium_ms_cm2": -1}, {"duration_ms": 0},
                                   {"delay_ms": -1}])
def test_reject_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match="Finite"):
        make_reference_cell(**kwargs)


def test_reference_converges_to_independent_ode_solution():
    with brainstate.environ.context(precision=64):
        coarse = make_reference_cell().run(dt=.005*u.ms, duration=30*u.ms)
        fine = make_reference_cell().run(dt=.00125*u.ms, duration=30*u.ms)
    vc = np.asarray(coarse.traces["voltage"].to_decimal(u.mV))
    vf = np.asarray(fine.traces["voltage"].to_decimal(u.mV))
    # Cell.run stamps the beginning of each step but records its end state.
    tf = np.arange(1, len(vf)+1)*.00125
    oracle = oracle_trace(tf)
    coarse_error = np.max(np.abs(vc-oracle[3::4]))
    fine_error = np.max(np.abs(vf-oracle))
    assert fine_error < .25  # mV, predeclared numerical waveform tolerance
    assert fine_error < coarse_error / 3
    assert np.sqrt(np.mean((vf-oracle)**2)) < .03
    assert abs(tf[vf.argmax()]-tf[oracle.argmax()]) <= .005
    assert abs(vf[-1]-oracle[-1]) < .002
