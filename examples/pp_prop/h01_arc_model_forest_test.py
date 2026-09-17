"""The fused ARC model's segmented sparse learner equals the per-cell learner."""
import braincell
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from braincell.network import pairs

import braintrace
from braintrace.datasets.h01_forest_cell import H01ForestCell
from braintrace.datasets.h01_forest_delivery import ForestContact
from braintrace.datasets.h01_forest_delivery_test import DELAY_MS, WEIGHT_US, _cells
from .h01_arc_model import H01ArcModel

IDS = ('a', 'b', 'c')
DT_MS = .005


def _percell_model():
    cells = _cells()
    network = braincell.Network(name='percell')
    for index, cell in enumerate(cells):
        network.add_population(f'cell_{index}', cell)
    for index, post in enumerate((1, 2)):
        network.add_edges(name=f'syn_{index}', pre='cell_0', post=f'cell_{post}', method=pairs([(0, 0)]))
        network.add_projection(name=f'syn_{index}', edges=f'syn_{index}', synapse=f'syn_{index}',
                               weight=WEIGHT_US*u.uS, delay=DELAY_MS*u.ms)
    return H01ArcModel(network, IDS, dt_ms=DT_MS, checkpoint_substeps=False)


def _forest_model():
    cells = _cells()
    for cell in cells:
        cell.init_state()
    forest = H01ForestCell(cells)
    forest.contacts = (ForestContact(0, 'syn_0', WEIGHT_US, DELAY_MS), ForestContact(0, 'syn_1', WEIGHT_US, DELAY_MS))
    network = braincell.Network(name='forest')
    network.add_population('forest', forest)
    return H01ArcModel(network, IDS, dt_ms=DT_MS, checkpoint_substeps=False)


def _learner(model):
    learner = braintrace.pp_prop.sparse(model, .9, max_bytes=2**30)
    learner.compile_graph(jnp.zeros(441))
    return learner


def _gradients(model, learner, events):
    """Parameter gradients of a squared-logit loss summed over ``events`` learner steps."""
    params = dict(input=model.input_weight, recurrent=model.recurrent_weight,
                  readout_weight=model.readout_weight, readout_bias=model.readout_bias)

    from braintrace._input_data import MultiStepData

    def loss(values):
        for name, state in params.items():
            state.value = values[name]
        soma = learner(MultiStepData(jnp.stack(events)))   # one window: (events, cells) soma voltages
        logits = jnp.tanh((soma+65.)/20.)*model._active() @ values['readout_weight']+values['readout_bias']
        return jnp.sum(jnp.mean(jnp.square(logits), axis=-1))

    values = {name: state.value for name, state in params.items()}
    grads = brainstate.transform.grad(loss, argnums=0)(values)
    return jax.tree.map(np.asarray, grads)


@pytest.fixture(scope='module')
def learners():
    with brainstate.environ.context(precision=64):
        percell = _percell_model()
        forest = _forest_model()
        return percell, _learner(percell), forest, _learner(forest)


def test_forest_layout_is_segmented_and_per_cell_sized(learners):
    percell, plearner, forest, flearner = learners
    p, f = plearner.graph.layout, flearner.graph.layout
    assert f.slots is not None and p.slots is None
    assert f.colors == p.colors   # same conflict structure: cell 0 alone, cells 1 and 2 each with their contact
    width = max(f.widths)
    assert width == 3 and f.elements == sum(int(np.prod(shape))*w for shape, w in zip(f.shapes, f.widths))
    assert f.elements < sum(int(np.prod(shape)) for shape in f.shapes)*len(f.colors)   # not one slot per output
    for table in f.slots:
        if table is not None and table.shape[-1] == width:
            rows = {tuple(r) for r in table.reshape(-1, width).tolist()}
            assert rows == {(0, -1, -1), (0, 1, 3), (0, 2, 4)}


def test_forest_gradients_equal_per_cell_gradients(learners):
    percell, plearner, forest, flearner = learners
    rng = np.random.default_rng(0)
    with brainstate.environ.context(precision=64):
        events = [jnp.asarray(rng.normal(size=441)*3., dtype=jnp.float64) for _ in range(2)]
        percell.reset_episode(plearner)
        forest.reset_episode(flearner)
        expected = _gradients(percell, plearner, events)
        got = _gradients(forest, flearner, events)
    for name in expected:
        scale = max(np.abs(expected[name]).max(), 1e-30)
        np.testing.assert_allclose(got[name], expected[name], rtol=0., atol=1e-9*scale,
                                   err_msg=name)
    assert any(np.abs(expected[name]).max() > 0. for name in ('input', 'recurrent'))


def _pattern(neurons, fanout=3):
    """Fixed encoder pattern over the first ``fanout`` neurons so that models of different width match."""
    targets = np.tile(np.arange(fanout, dtype=np.int32), 441)
    indptr = np.arange(442, dtype=np.int32)*fanout
    return targets, indptr


def _static_model(slots, capacity):
    from braintrace.datasets.h01_forest_contacts_test import _cells as shared_cells
    cells = shared_cells()
    for cell in cells:
        cell.init_state()
    forest = H01ForestCell(cells, slots=slots)
    # A chain 0 -> 1 -> 2: the segment of cell 2 is reached only through cell 1's segment.
    forest.contact_spec = dict(capacity=capacity, max_delay_ms=DELAY_MS, rows=[
        dict(pre=0, post=1, kind=0, weight_us=WEIGHT_US, delay_ms=DELAY_MS),
        dict(pre=1, post=2, kind=1, weight_us=WEIGHT_US, delay_ms=DELAY_MS)])
    network = braincell.Network(name='forest')
    network.add_population('forest', forest)
    return H01ArcModel(network, IDS, dt_ms=DT_MS, checkpoint_substeps=False, input_pattern=_pattern(3*slots))


def _percell_shared_model():
    from braintrace.datasets.h01_forest_contacts_test import _cells as shared_cells
    cells = shared_cells()
    network = braincell.Network(name='percell')
    for index, cell in enumerate(cells):
        network.add_population(f'cell_{index}', cell)
    for name, kind, pre, post in (('e', 'contact_exc', 0, 1), ('i', 'contact_inh', 1, 2)):
        network.add_edges(name=name, pre=f'cell_{pre}', post=f'cell_{post}', method=pairs([(0, 0)]))
        network.add_projection(name=name, edges=name, synapse=kind, weight=WEIGHT_US*u.uS, delay=DELAY_MS*u.ms)
    return H01ArcModel(network, IDS, dt_ms=DT_MS, checkpoint_substeps=False, input_pattern=_pattern(3))


def _align(reference, wide):
    """Copy the reference parameters into the wide (slotted) model's leading rows."""
    wide.input_weight.value = reference.input_weight.value
    wide.readout_bias.value = reference.readout_bias.value
    wide.readout_weight.value = wide.readout_weight.value.at[:3].set(reference.readout_weight.value)
    wide.recurrent_weight.value = wide.recurrent_weight.value.at[:2].set(reference.recurrent_weight.value)
    wide.contact_magnitude.value = jnp.abs(wide.recurrent_weight.value)


@pytest.fixture(scope='module')
def static_pair():
    with brainstate.environ.context(precision=64):
        reference = _percell_shared_model()
        wide = _static_model(2, 4)
        _align(reference, wide)
        return reference, _learner(reference), wide, _learner(wide)


def test_static_model_forward_and_gradients_equal_per_cell(static_pair):
    reference, rlearner, wide, wlearner = static_pair
    rng = np.random.default_rng(1)
    with brainstate.environ.context(precision=64):
        # 24 events = 2.4 ms: cell 0 crosses 0 mV at event 13 (1.3 ms), so the 0.5 ms contact reaches
        # cell 1 at 1.8 ms; at 16 events the spike sits in the ring buffer undelivered and the
        # contact magnitudes carry no credit (recurrent gradient exactly zero on both models).
        events = [jnp.asarray(rng.normal(size=441)*3., dtype=jnp.float64) for _ in range(24)]
        reference.reset_episode(rlearner)
        wide.reset_episode(wlearner)
        soma_r = np.asarray(brainstate.transform.for_loop(lambda e: reference.update(e), jnp.stack(events)))
        soma_w = np.asarray(brainstate.transform.for_loop(lambda e: wide.update(e), jnp.stack(events)))
        np.testing.assert_allclose(soma_w[:, :3], soma_r, rtol=0., atol=1e-10)
        reference.reset_episode(rlearner)
        wide.reset_episode(wlearner)
        expected = _gradients(reference, rlearner, events)
        got = _gradients(wide, wlearner, events)
    scale = lambda name: max(np.abs(expected[name]).max(), 1e-30)   # noqa: E731
    np.testing.assert_allclose(got['input'], expected['input'], rtol=0., atol=1e-9*scale('input'))
    np.testing.assert_allclose(got['readout_bias'], expected['readout_bias'], rtol=0., atol=1e-9*scale('readout_bias'))
    np.testing.assert_allclose(got['readout_weight'][:3], expected['readout_weight'], rtol=0., atol=1e-9*scale('readout_weight'))
    assert np.abs(got['readout_weight'][3:]).max() == 0.   # dormant slots carry no credit
    np.testing.assert_allclose(got['recurrent'][:2], expected['recurrent'], rtol=0., atol=1e-9*max(scale('recurrent'), 1e-12))
    assert np.abs(got['recurrent'][2:]).max() == 0.
    assert np.abs(expected['recurrent']).max() > 0.   # the chain delivered: contact magnitudes carry credit
    layout = wlearner.graph.layout
    assert layout.slots is not None and layout.color_count <= rlearner.graph.layout.color_count


def test_static_model_clone_and_contact_in_place(static_pair):
    reference, rlearner, wide, wlearner = static_pair
    with brainstate.environ.context(precision=64):
        step = brainstate.transform.jit(wide.update)
        event = jnp.zeros(441)
        for _ in range(2):   # the second call re-keys on device-placed state values from the first
            jax.block_until_ready(step(event))
        clone = wide.clone(0)
        row = wide.add_contact(clone, 2, kind=0, weight_us=.03)
        assert clone == 3 and row == 2
        assert wide.sparse_structure_signature() == ((0, 0, 1, 0), (1, 1, 2, 1), (2, 3, 2, 0))
        import logging
        records = []

        class Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())
        handler = Handler()
        logging.getLogger().addHandler(handler)
        with jax.log_compiles(True):
            jax.block_until_ready(step(event))
        logging.getLogger().removeHandler(handler)
        assert not [m for m in records if 'ompil' in m]
        with pytest.raises(ValueError, match='No dormant slot'):
            wide.clone(0)
