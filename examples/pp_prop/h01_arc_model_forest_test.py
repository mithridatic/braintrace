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

    def loss(values):
        for name, state in params.items():
            state.value = values[name]
        total = 0.
        for event in events:
            learner(event)
            total = total+jnp.mean(jnp.square(model.readout()))
        return total

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
