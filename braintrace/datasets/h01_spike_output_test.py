"""One output event per selected site, independent of other compartment crossings."""
import braincell
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell.filter import RootLocation
from .h01_spike_output import restrict_spike_output


def cell():
    branch = braincell.Branch.from_lengths(lengths=[30.]*u.um, radii=[2., 2.]*u.um, type="soma")
    return braincell.Cell(braincell.Morphology.from_root(branch, name="soma"),
        pop_size=(1,), cv_policy=braincell.MaxCVLen(10.*u.um), V_th=0.*u.mV)


def test_propagating_crossings_outside_site_do_not_emit():
    model = cell()
    site = restrict_spike_output(model, RootLocation(.5))
    index = site["output_cv_id"]
    before = np.full((1, model.n_cv), -10.)*u.mV
    after = np.full((1, model.n_cv), 10.)*u.mV
    event = brainstate.transform.jit(model.get_spike)(before, after)
    assert np.count_nonzero(event) == 1 and event[0, index] == 1
    values = np.full((1, model.n_cv), 10.)
    values[0, index] = -10.
    event = brainstate.transform.jit(model.get_spike)(before, values*u.mV)
    assert not np.any(event)
    assert restrict_spike_output(model, RootLocation(.5)) == site


def test_multiple_source_sites_and_changed_mesh_are_rejected():
    model = cell()
    with pytest.raises(ValueError, match="single source"):
        restrict_spike_output(model, RootLocation(.2) | RootLocation(.8))
    restrict_spike_output(model, RootLocation(.5))
    with pytest.raises(ValueError, match="mesh"):
        model.spk_fun(np.ones((1, model.n_cv+1)))


def test_output_mask_survives_compiled_cell_run():
    with brainstate.environ.context(precision=64):
        model = cell()
        model.place(RootLocation(.5), braincell.mech.StateProbe(field="v", name="voltage"))
        restrict_spike_output(model, RootLocation(.5))
        model.run(dt=.01*u.ms, duration=.02*u.ms)
        assert np.asarray(model.spike.value).shape == (1, model.n_cv)
        assert not np.any(model.spike.value)


def test_probe_at_selected_midpoint_preserves_output_mesh():
    model = cell()
    site = restrict_spike_output(model, RootLocation(.5))
    before = tuple((cv.id, cv.branch_id, cv.prox, cv.dist) for cv in model.cvs)
    model.place(braincell.filter.AtLocation(*site["output_midpoint"]),
                braincell.mech.StateProbe(field="v", name="output_voltage"))
    after = tuple((cv.id, cv.branch_id, cv.prox, cv.dist) for cv in model.cvs)
    assert before == after
    assert site == restrict_spike_output(model, RootLocation(.5))
