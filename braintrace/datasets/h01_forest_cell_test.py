"""The forest cell integrates exactly what its source cells integrate."""
import braincell
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell.filter import AllRegion, BranchSlice, RootLocation
from braincell.mech import Channel, Ion, StateProbe

from . import h01_calcium_solver, h01_pv_channels  # noqa: F401  registers the integrator and channels
from .h01_construction import H01Cell
from .h01_forest_cell import H01ForestCell
from .h01_spike_output import restrict_spike_output


def _morphology(dendrite_um):
    soma = braincell.Soma(lengths=np.array([12.])*u.um, radii_proximal=np.array([6.])*u.um,
                          radii_distal=np.array([6.])*u.um)
    morpho = braincell.Morphology(root_name='soma', root_branch=soma)
    dendrite = braincell.Branch(lengths=np.array([dendrite_um])*u.um, radii_proximal=np.array([1.])*u.um,
                                radii_distal=np.array([.8])*u.um, type='dendrite')
    morpho.attach(parent='soma', child_branch=dendrite, child_name='dend')
    return morpho


def make_cell(seed, *, dendrite_um=40., na=1.2, k=.6, current_na=.25, m_open=1., calcium=True):
    """A two-branch cell with donor-like densities; ``seed`` only labels it."""
    cell = H01Cell(_morphology(dendrite_um), cv_policy=braincell.MaxCVLen(10.*u.um), V_init=-70.*u.mV,
                   solver='h01_staggered_calcium_implicit', pop_size=(1,), name=f'cell{seed}')
    cell.paint(AllRegion(), braincell.CableProperty(membrane_capacitance=1.*u.uF/u.cm**2,
               axial_resistivity=100.*u.ohm*u.cm, resting_potential=-70.*u.mV))
    cell.paint(AllRegion(), Channel('IL', name='leak', g_max=.05*u.mS/u.cm**2, E=-70.*u.mV))
    cell.paint(AllRegion(), Ion('SodiumFixed', name='sodium', E=50.*u.mV))
    cell.paint(AllRegion(), Ion('PotassiumFixed', name='potassium', E=-85.*u.mV))
    soma = BranchSlice(0, prox=0., dist=1.)
    cell.paint(soma, Channel('H01PV_NaTg', name='pv_NaTg', g_max=na*100.*u.mS/u.cm**2, m_open=m_open))
    cell.paint(soma, Channel('H01PV_Kv3_1', name='pv_Kv3_1', g_max=k*100.*u.mS/u.cm**2))
    cell.paint(AllRegion(), Channel('H01PV_Ih', name='pv_Ih', g_max=.01*u.mS/u.cm**2))
    if calcium:
        cell.paint(soma, Ion('H01PV_Calcium', name='calcium', decay=80.*u.ms, gamma=.001))
        cell.paint(soma, Channel('H01PV_Ca_HVA', name='pv_Ca_HVA', g_max=.5*u.mS/u.cm**2))
        cell.paint(soma, Channel('H01PV_SK', name='pv_SK', g_max=1.*u.mS/u.cm**2))
    cell.place(RootLocation(.5), StateProbe(field='v', name='voltage'))
    cell.place(RootLocation(.5), braincell.CurrentClamp(delay=.2*u.ms, durations=3.*u.ms, amplitudes=current_na*u.nA))
    restrict_spike_output(cell, RootLocation(.5))
    return cell


def _run(step, cells, steps, dt_ms):
    def body(_):
        with brainstate.environ.context(dt=dt_ms*u.ms):
            step()
        return tuple(c.V.value.to_decimal(u.mV)[0] for c in cells), tuple(np.asarray(c.spike.value)[0] for c in cells)
    return brainstate.transform.for_loop(body, np.arange(steps))


@pytest.fixture(scope='module')
def pair():
    with brainstate.environ.context(precision=64):
        cells = (make_cell(0, na=1.2, k=.6, m_open=.5), make_cell(1, dendrite_um=25., na=.9, k=.9, current_na=.3, calcium=False))
        for cell in cells:
            cell.init_state()
        forest = H01ForestCell(cells)
        forest.init_state()
    return cells, forest


def test_forest_has_one_layout_per_mechanism(pair):
    cells, forest = pair
    keys = {(m.category, m.class_name, m.instance_name) for c in cells for m in
            (c._runtime.layout_mechanisms[l.id] for l in c._runtime.layouts if l.target == 'density')}
    density = [l for l in forest._runtime.layouts if l.target == 'density']
    assert len(density) == len(keys)
    assert forest.n_cv == sum(c.n_cv for c in cells)
    assert forest.forest_offsets.cv.tolist() == [0, cells[0].n_cv, cells[0].n_cv+cells[1].n_cv]
    assert len(forest._runtime.clamp_routing_table.midpoint_ids) == 2


def test_forest_parameters_are_the_source_parameters(pair):
    cells, forest = pair
    node = next(n for l in forest._runtime.layouts for n in [forest._runtime.runtime_nodes.get(l.id)]
                if n is not None and getattr(n, 'mechanism', None) == 'NaTg')
    offsets = forest.forest_offsets.point
    g = np.asarray(node.g_max.to_decimal(u.mS/u.cm**2))[0]
    for index, cell in enumerate(cells):
        source = next(n for l in cell._runtime.layouts for n in [cell._runtime.runtime_nodes.get(l.id)]
                      if n is not None and getattr(n, 'mechanism', None) == 'NaTg')
        np.testing.assert_array_equal(g[offsets[index]:offsets[index+1]], np.asarray(source.g_max.to_decimal(u.mS/u.cm**2))[0])
    opening = np.asarray(node.phase_factors['m'][0])[0]
    assert opening[offsets[0]:offsets[1]].max() == .5 and opening[offsets[1]:offsets[2]].min() == 1.


def test_forest_voltages_equal_per_cell_voltages(pair):
    cells, forest = pair
    steps, dt = 400, .005
    with brainstate.environ.context(precision=64):
        def per_cell():
            for cell in cells:
                cell._update_dynamics()
        expected, expected_spikes = _run(per_cell, cells, steps, dt)
        got, got_spikes = _run(forest._update_dynamics, (forest,), steps, dt)
    got = np.asarray(got[0])
    for index, cell in enumerate(cells):
        lo, hi = forest.forest_offsets.cv[index], forest.forest_offsets.cv[index+1]
        np.testing.assert_allclose(got[:, lo:hi], np.asarray(expected[index]), rtol=0., atol=1e-9)
        out = forest.output_cv_ids[index]
        np.testing.assert_array_equal(np.asarray(got_spikes[0])[:, out], np.asarray(expected_spikes[index])[:, cell.spk_fun.cv_id])
    assert np.asarray(expected[0]).max() > 0.   # the window contains a spike


def test_forest_rejects_uninitialized_sources():
    with brainstate.environ.context(precision=64), pytest.raises(ValueError, match='initialized'):
        H01ForestCell((make_cell(5),))
