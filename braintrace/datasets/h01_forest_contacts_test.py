"""Static-capacity forest: dormant slots and a fixed contact table, no shape changes."""
import logging

import braincell
import brainstate
import brainunit as u
import jax
import numpy as np
import pytest
from braincell.filter import RootLocation
from braincell.mech import Synapse
from braincell.network import pairs

from .h01_forest_cell import H01ForestCell
from .h01_forest_cell_test import make_cell
from .h01_forest_contacts import KINDS, ForestContactTable, contact_kind
from .h01_network_step import H01NetworkStep

DT_MS = .005
WEIGHT_US, DELAY_MS = .02, .5


def _cells():
    cells = (make_cell(0, na=1.2, k=.6), make_cell(1, dendrite_um=25., current_na=.05, calcium=False),
             make_cell(2, dendrite_um=30., current_na=0.))
    for cell in cells:
        for kind in KINDS.values():
            cell.place(RootLocation(.5), Synapse('ExpSyn', name=kind['name'], e=kind['reversal_mv']*u.mV,
                                                 tau=kind['tau_ms']*u.ms, weight=1.*u.uS))
    return cells


def _record(step, forest, steps):
    def body(index):
        with brainstate.environ.context(t=index*DT_MS*u.ms):
            step.update(sample_probes=False)
        return forest.V.value.to_decimal(u.mV)[0]
    return np.asarray(brainstate.transform.for_loop(body, np.arange(steps)))


def _percell(steps):
    cells = _cells()
    network = braincell.Network(name='percell')
    for index, cell in enumerate(cells):
        network.add_population(f'cell_{index}', cell)
    for name, kind, post in (('e', 'contact_exc', 1), ('i', 'contact_inh', 2)):
        network.add_edges(name=name, pre='cell_0', post=f'cell_{post}', method=pairs([(0, 0)]))
        network.add_projection(name=name, edges=name, synapse=kind, weight=WEIGHT_US*u.uS, delay=DELAY_MS*u.ms)
    step = H01NetworkStep(network, DT_MS)
    def body(index):
        with brainstate.environ.context(t=index*DT_MS*u.ms):
            step.update(sample_probes=False)
        return tuple(c.V.value.to_decimal(u.mV)[0] for c in cells)
    return [np.asarray(v) for v in brainstate.transform.for_loop(body, np.arange(steps))]


def _static(slots, capacity, contacts):
    cells = _cells()
    for cell in cells:
        cell.init_state()
    forest = H01ForestCell(cells, slots=slots)
    forest.contact_spec = dict(capacity=capacity, max_delay_ms=DELAY_MS, rows=[
        dict(pre=pre, post=post, kind=kind, weight_us=WEIGHT_US, delay_ms=DELAY_MS) for pre, post, kind in contacts])
    network = braincell.Network(name='forest')
    network.add_population('forest', forest)
    return forest, H01NetworkStep(network, DT_MS)


@pytest.fixture(scope='module')
def runs():
    steps = 500
    with brainstate.environ.context(precision=64):
        percell = _percell(steps)
        forest, step = _static(2, 4, [(0, 1, 0), (0, 2, 1)])
        fused = _record(step, forest, steps)
    return percell, forest, step, fused


def test_kind_lookup():
    assert contact_kind(0., 2.) == 0 and contact_kind(-80., 5.) == 1
    with pytest.raises(ValueError):
        contact_kind(-70., 2.)


def test_static_contacts_equal_per_cell_projections(runs):
    percell, forest, step, fused = runs
    assert forest.cells == 6 and forest.slots == 2
    for index in range(3):
        lo, hi = forest.forest_offsets.cv[index], forest.forest_offsets.cv[index+1]
        np.testing.assert_allclose(fused[:, lo:hi], percell[index], rtol=0., atol=1e-12)
    assert percell[0].max() > 0.
    table = step.contact_table
    assert table.capacity == 4 and sorted(table.rows) == [0, 1]
    assert np.asarray(table.active.value).tolist() == [1., 1., 0., 0.]
    assert table.post_cell(forest, 0) == 1 and table.post_cell(forest, 1) == 2


def test_dormant_slots_are_exact_copies_and_masked(runs):
    percell, forest, step, fused = runs
    for source in range(3):
        slot = forest.slot_of(source, 1)
        lo, hi = forest.forest_offsets.cv[slot], forest.forest_offsets.cv[slot+1]
        base = forest.forest_offsets.cv[source]
        # A dormant copy of cell 0 receives no contact and is not a contact source, so it
        # follows cell 0 (whose only inputs are its clamp) exactly; copies of 1 and 2 miss
        # the delivered contacts and differ from their sources.
        if source == 0:
            np.testing.assert_allclose(fused[:, lo:hi], fused[:, base:base+hi-lo], rtol=0., atol=1e-12)
    spikes = np.asarray(forest.spike.value)[0]
    assert np.asarray(forest.active.value).tolist() == [1., 1., 1., 0., 0., 0.]


def test_clone_and_contact_in_place_do_not_recompile(runs):
    percell, forest, step, fused = runs
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    handler = Handler()
    logging.getLogger().addHandler(handler)
    with brainstate.environ.context(precision=64), jax.log_compiles(True):
        advance = brainstate.transform.jit(lambda: step.update(sample_probes=False))
        with brainstate.environ.context(t=0.*u.ms):
            jax.block_until_ready(advance())
        compiled_before = len([m for m in records if 'ompil' in m])
        shapes_before = jax.tree.map(lambda x: x.shape, {k: v.value for k, v in brainstate.graph.states(forest).items()})
        slot = forest.free_slot(0)
        forest.activate(slot)
        row = step.contact_table.free_row()
        point = int(np.asarray(forest.runtime.node_tree.cv_to_mid_node_id)[forest.soma_cv_ids[slot]])
        step.contact_table.write(row, pre=1, post_point=point, kind=0, weight_us=WEIGHT_US, delay_ms=DELAY_MS)
        with brainstate.environ.context(t=DT_MS*u.ms):
            jax.block_until_ready(advance())
        compiled_after = len([m for m in records if 'ompil' in m])
        shapes_after = jax.tree.map(lambda x: x.shape, {k: v.value for k, v in brainstate.graph.states(forest).items()})
    logging.getLogger().removeHandler(handler)
    assert compiled_before >= 1 and compiled_after == compiled_before
    assert shapes_after == shapes_before
    assert np.asarray(forest.active.value)[slot] == 1. and step.contact_table.rows[row]['pre'] == 1
    with pytest.raises(ValueError, match='outside'):
        step.contact_table.write(3, pre=0, post_point=point, kind=0, weight_us=.01, delay_ms=5.)
    with pytest.raises(ValueError, match='carries no'):
        step.contact_table.write(3, pre=0, post_point=0, kind=0, weight_us=.01, delay_ms=.1)
