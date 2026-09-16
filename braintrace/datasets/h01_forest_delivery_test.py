"""Contacts delivered inside the forest equal BrainCell's per-population delivery."""
import braincell
import brainstate
import brainunit as u
import numpy as np
import pytest
from braincell.filter import RootLocation
from braincell.mech import Synapse
from braincell.network import pairs

from .h01_forest_cell import H01ForestCell
from .h01_forest_cell_test import make_cell
from .h01_forest_delivery import ForestContact, forest_delivery, forest_presynaptic_spikes
from .h01_network_step import H01NetworkStep

DT_MS = .005
WEIGHT_US, DELAY_MS = .02, .5


def _cells():
    """Cell 0 fires (its clamp) and contacts cell 1 (weak clamp) and cell 2 (none)."""
    cells = (make_cell(0, na=1.2, k=.6), make_cell(1, dendrite_um=25., current_na=.05, calcium=False),
             make_cell(2, dendrite_um=30., current_na=0.))
    for index, (cell, reversal, tau) in enumerate(((cells[1], 0., 2.), (cells[2], -80., 5.))):
        cell.place(RootLocation(.5), Synapse('ExpSyn', name=f'syn_{index}', e=reversal*u.mV, tau=tau*u.ms, weight=1.*u.uS))
    return cells


def _record(step, forest_or_cells, steps):
    def body(index):
        with brainstate.environ.context(t=index*DT_MS*u.ms):
            step.update(sample_probes=False)
        return tuple(c.V.value.to_decimal(u.mV)[0] for c in forest_or_cells)
    return brainstate.transform.for_loop(body, np.arange(steps))


@pytest.fixture(scope='module')
def traces():
    with brainstate.environ.context(precision=64):
        cells = _cells()
        network = braincell.Network(name='percell')
        for index, cell in enumerate(cells):
            network.add_population(f'cell_{index}', cell)
        for index, post in enumerate((1, 2)):
            network.add_edges(name=f'syn_{index}', pre='cell_0', post=f'cell_{post}', method=pairs([(0, 0)]))
            network.add_projection(name=f'syn_{index}', edges=f'syn_{index}', synapse=f'syn_{index}',
                                   weight=WEIGHT_US*u.uS, delay=DELAY_MS*u.ms)
        percell = _record(H01NetworkStep(network, DT_MS), cells, 500)
        fused_cells = _cells()
        for cell in fused_cells:
            cell.init_state()
        forest = H01ForestCell(fused_cells)
        forest.contacts = (ForestContact(0, 'syn_0', WEIGHT_US, DELAY_MS), ForestContact(0, 'syn_1', WEIGHT_US, DELAY_MS))
        fused_network = braincell.Network(name='forest')
        fused_network.add_population('forest', forest)
        fused = np.asarray(_record(H01NetworkStep(fused_network, DT_MS), (forest,), 500)[0])
        silent_cells = _cells()
        for cell in silent_cells:
            cell.init_state()
        silent = H01ForestCell(silent_cells)
        silent_network = braincell.Network(name='silent')
        silent_network.add_population('forest', silent)
        no_contact = np.asarray(_record(H01NetworkStep(silent_network, DT_MS), (silent,), 500)[0])
    return cells, [np.asarray(v) for v in percell], forest, fused, no_contact


def test_forest_blocks_target_each_contact_synapse(traces):
    cells, _, forest, _, _ = traces
    blocks, ops = forest_delivery(forest, forest.contacts, dt_ms=DT_MS)
    assert len(blocks) == len(ops) == 2
    assert [int(b.delay_steps) for b in blocks] == [100, 100]
    assert [int(b.pre_index[0]) for b in blocks] == [0, 0]
    assert len({b.source.layout_id for b in blocks}) == 2
    assert all(b.source.n_active == 1 and int(b.flat_target_index[0]) == 0 for b in blocks)


def test_forest_delivery_equals_per_cell_delivery(traces):
    cells, percell, forest, fused, no_contact = traces
    for index, cell in enumerate(cells):
        lo, hi = forest.forest_offsets.cv[index], forest.forest_offsets.cv[index+1]
        np.testing.assert_allclose(fused[:, lo:hi], percell[index], rtol=0., atol=1e-9)
    assert percell[0].max() > 0.   # the pre cell spiked
    for index in (1, 2):   # the contacts changed the post cells: delivery is live on both paths
        lo, hi = forest.forest_offsets.cv[index], forest.forest_offsets.cv[index+1]
        assert np.abs(fused[:, lo:hi]-no_contact[:, lo:hi]).max() > 1e-3
    lo, hi = forest.forest_offsets.cv[0], forest.forest_offsets.cv[1]
    np.testing.assert_allclose(fused[:, lo:hi], no_contact[:, lo:hi], rtol=0., atol=1e-9)   # nothing targets cell 0


def test_presynaptic_spikes_are_per_cell_output_sites(traces):
    _, _, forest, _, _ = traces
    with brainstate.environ.context(precision=64):
        spike = np.zeros(forest.spike.value.shape)
        spike[0, forest.output_cv_ids[1]] = 1.
        spike[0, forest.forest_offsets.cv[0]] = 1.   # a non-output CV never counts
        forest.spike.value = spike
        assert forest_presynaptic_spikes(forest).tolist() == [False, True, False]
