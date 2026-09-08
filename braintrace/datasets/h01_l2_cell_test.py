"""Donor geometry reference initialization, explicit source mode, mesh policy, and segments."""
import json
from pathlib import Path
import braincell
import brainstate
import brainunit as u
import numpy as np
import pytest
from . import h01_l2_cell
from .h01_l2_cell import make_l2_cell, _stimulus_segments
from .h01_pv_morphology import make_pv_morphology
REFERENCE = json.loads((Path(__file__).resolve().parents[2]/"docs/evidence/h01-l2-geometry-reference.json").read_text())


@pytest.mark.parametrize("mode", ["source", "candidate", "b3"])
def test_donor_cell_runs_with_finite_calcium(mode):
    with brainstate.environ.context(precision=64):
        result = make_l2_cell(REFERENCE, mode=mode).run(dt=.001*u.ms, duration=.05*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
    assert (np.asarray(result.traces["calcium"].to_decimal(u.mM)) > 0).all()


@pytest.mark.parametrize("kwargs", [{"current_na": np.nan}, {"duration_ms": 0.}, {"delay_ms": -1.},
                                    {"max_cv_length_um": 0.}, {"current_na": [.1, .2], "duration_ms": 3.},
                                    {"current_na": [], "duration_ms": []},
                                    {"current_na": [.1, .2], "duration_ms": [1., -1.]}])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError, match="Finite current"):
        make_l2_cell(REFERENCE, **kwargs)


def test_segments_are_returned_as_equal_length_arrays():
    amplitudes, durations = _stimulus_segments([0., .31, 0.], [1020., 1000., 80.])
    assert amplitudes.tolist() == [0., .31, 0.] and durations.tolist() == [1020., 1000., 80.]


def test_explicit_cv_policy_replaces_max_cv_length(monkeypatch):
    seen = {}
    real_cell = h01_l2_cell.H01Cell

    def spy(morphology, **kwargs):
        seen.update(kwargs)
        return real_cell(morphology, **kwargs)
    monkeypatch.setattr(h01_l2_cell, "H01Cell", spy)
    counts = tuple(3 for _ in make_pv_morphology(REFERENCE).branches)
    policy = braincell.CVPerBranchList(counts)
    make_l2_cell(REFERENCE, cv_policy=policy)
    assert seen["cv_policy"] is policy
    make_l2_cell(REFERENCE, max_cv_length_um=7.)
    assert isinstance(seen["cv_policy"], braincell.MaxCVLen)


def test_segment_protocol_runs_with_copied_mesh():
    counts = tuple(1 for _ in make_pv_morphology(REFERENCE).branches)
    with brainstate.environ.context(precision=64):
        cell = make_l2_cell(REFERENCE, current_na=[.05, 0., .31], duration_ms=[.01, .01, .01],
                            delay_ms=0., cv_policy=braincell.CVPerBranchList(counts))
        result = cell.run(dt=.001*u.ms, duration=.05*u.ms)
    assert np.isfinite(result.traces["voltage"].to_decimal(u.mV)).all()
