"""Finite-window sparse pp-prop through shared chemical feedback."""

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np

from braintrace._algorithm.sparse_pp_prop import SparsePPProp
from braintrace._testing.oracle import chunked_online_param_gradients
from .environment import ChemicalEnvironment, MembraneMap
from .transport import DiffusionGraph


class ChemicalLearning(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = brainstate.ParamState(jnp.array([[.01, .02], [.03, .04]]))
        graph = DiffusionGraph([2., 3.], [[0, 1]], [.3], dt_ms=.005)
        self.environment = ChemicalEnvironment(graph, graph, MembraneMap([0, 1], [10., 10.], 2), [140., 140.])

    def update(self, x):
        current = braintrace.matmul(x, self.weight.value)
        # Chemical reversal feeds back into the next physical current, as it
        # does for a cable channel; rates and geometry remain nontrainable.
        current = current*(self.environment.reversal_mv()+120.)/20.
        potassium, _ = self.environment.update(current, jnp.zeros(2))
        return (potassium-2.5)*100.


def test_finite_window_chemical_feedback_gradients_depend_on_trace_decay_and_descend():
    with brainstate.environ.context(precision=64):
        inputs = jnp.array([[.2, .3], [.5, .1], [.1, .4]])
        captured = {}
        def gradients(decay):
            return chunked_online_param_gradients(ChemicalLearning, inputs,
                algo_factory=lambda model: SparsePPProp(model, decay), chunk_size=1, compiled_scan=True,
                initialize_model=False, after_init=lambda m, a: captured.update(model=m, learner=a))
        low, high = gradients(.2), gradients(.8)
        a, b = low[('weight',)], high[('weight',)]
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(b) > 0
        assert not np.allclose(a, b, rtol=1e-4, atol=1e-12)
        model = captured['model']
        def evaluate():
            model.environment.reset_state()
            return jnp.square(brainstate.transform.for_loop(model.update, inputs)).sum()
        evaluate = brainstate.transform.jit(evaluate)
        before = evaluate()
        model.weight.value = model.weight.value-1e-3*b/jnp.max(jnp.abs(b))
        assert evaluate() < before
        assert model.environment.valid.value
        captured['learner'].reset_state()
        assert all(not np.any(f) for f in captured['learner'].factors.value)
