"""ETP graph extraction retains heterogeneous cable-like state dependencies."""

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np

from .sparse_io_graph import SparseIOGraph


def test_typed_random_key_retains_forward_state_without_temporal_factor():
    class Stochastic(_CableBlocks):
        def __init__(self):
            super().__init__()
            self.rng = brainstate.random.RandomState(21)

        def update(self, x):
            return super().update(x) * self.rng.bernoulli(.36, size=(2,))

    model = Stochastic()
    x = jnp.ones(2)
    graph = SparseIOGraph(model, x)
    assert all(path[-1] != 'rng' for path in graph.state_paths)
    raw = graph.inputs(x)
    key_indices = [i for i, value in enumerate(raw)
                   if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)]
    assert len(key_indices) == 1
    full, _, _ = graph.forward(raw)
    output_keys = [value for value in full
                   if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)]
    assert len(output_keys) == 1
    assert not np.array_equal(jax.random.key_data(output_keys[0]),
                              jax.random.key_data(raw[key_indices[0]]))


class _CableBlocks(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = brainstate.ParamState(jnp.array([[.2, .3], [.4, .5]]))
        self.a = brainstate.HiddenState(jnp.zeros(3))
        self.b = brainstate.HiddenState(jnp.zeros(5))

    def update(self, x):
        y = braintrace.matmul(x, self.weight.value)
        self.a.value = jnp.tanh(.5*self.a.value + y[0])
        self.b.value = jnp.tanh(.3*self.b.value + y[1])
        return jnp.array([self.a.value[0], self.b.value[0]])


def test_graph_preserves_forward_and_indexed_parameter_support():
    model = _CableBlocks()
    x = jnp.array([.1, -.2])
    graph = SparseIOGraph(model, x)
    assert graph.layout.outputs == ((0,), (1,))
    assert graph.layout.elements == 8
    raw = graph.inputs(x)
    full, xs, y = graph.forward(raw)
    hidden = tuple(raw[i] for i in graph.state_input_indices)
    result = graph.transition(y, hidden, raw)
    for value, index in zip(result, graph.state_output_indices):
        np.testing.assert_allclose(value, full[index], rtol=1e-6)
    np.testing.assert_allclose(xs[0], x)
    np.testing.assert_allclose(full[0], model(x))


def test_overwritten_drive_does_not_join_disconnected_temporal_factors():
    class Driven(_CableBlocks):
        def __init__(self):
            super().__init__()
            self.drive = brainstate.ShortTermState(jnp.zeros(2))

        def update(self, x):
            self.drive.value = braintrace.matmul(x, self.weight.value)
            self.a.value = jnp.tanh(.5*self.a.value+self.drive.value[0])
            self.b.value = jnp.tanh(.3*self.b.value+self.drive.value[1])
            return self.a.value[0]+self.b.value[0]
    graph = SparseIOGraph(Driven(), jnp.ones(2))
    assert graph.layout.color_count == 1
    assert graph.layout.elements == 8
    assert all(path[-1] != 'drive' for path in graph.state_paths)


def test_actual_cable_graph_keeps_voltage_and_contact_queue(tmp_path):
    from braintrace.datasets.h01_network_step_test import _contact_network
    from examples.pp_prop.h01_arc_model import H01ArcModel
    model = H01ArcModel(_contact_network(tmp_path), ['5805562981', '5965472721'])
    graph = SparseIOGraph(model, jnp.zeros(441))
    assert len(graph.operations) == 2
    assert any('ring_buffers' in path for path in graph.state_paths)
    assert any(path[-1] == 'V' for path in graph.state_paths)
    assert graph.layout.elements > 0
    assert graph.layout.nbytes < 1024**2
