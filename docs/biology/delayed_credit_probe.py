"""Independent closed-form diagnosis of weight-only delayed pp-prop credit."""

import brainstate
import braintrace
import jax.numpy as jnp
import numpy as np

from braintrace._testing.oracle import chunked_online_param_gradients


class _Delay(brainstate.nn.Module):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay
        self.weight = brainstate.ParamState(jnp.array([.5, -.7]))
        self.queue = brainstate.HiddenState(jnp.zeros((max(1, delay), 2)))

    def update(self, impulse):
        value = braintrace.element_wise(self.weight.value, weight_fn=jnp.abs)*impulse
        output = self.queue.value[0] if self.delay else value
        self.queue.value = jnp.concatenate((self.queue.value[1:], value[None]), axis=0)
        return output


def measure(delay, emission, decay):
    """Compare a delayed unit impulse with its analytic gradient and trace law.

    Parameters
    ----------
    delay : int
        Nonnegative number of event boundaries before arrival.
    emission : int
        Nonnegative zero-based impulse event index.
    decay : float
        Output and input smoothing coefficient in [0, 1).

    Returns
    -------
    dict
        Analytic, finite-window, BPTT and finite-difference derivatives.
    """
    if delay < 0 or emission < 0 or not 0 <= decay < 1:
        raise ValueError('Require nonnegative delay/emission and decay in [0, 1)')
    inputs = jnp.zeros((emission+delay+2, 2)).at[emission].set(1.)
    factory = lambda: _Delay(delay)
    actual = chunked_online_param_gradients(factory, inputs,
        algo_factory=lambda model: braintrace.pp_prop.sparse(model, decay),
        chunk_size=1, compiled_scan=True, initialize_model=False)[('weight',)]
    model = factory()

    def objective():
        model.queue.value = jnp.zeros_like(model.queue.value)
        return jnp.square(brainstate.transform.for_loop(model, inputs)).sum()

    reference = brainstate.transform.jit(brainstate.transform.grad(
        objective, {'weight': model.weight}))()['weight']
    base = model.weight.value
    direction = jnp.array([.6, -.8])
    evaluate = brainstate.transform.jit(objective)
    model.weight.value = base+1e-5*direction
    plus = evaluate()
    model.weight.value = base-1e-5*direction
    minus = evaluate()
    exact = 2*np.asarray(base)
    ratio = 1. if delay == 0 else (1-decay)*decay**(delay-1)/(1-decay**(emission+delay))
    return dict(delay=delay, emission=emission, decay=decay, exact_gradient=exact.tolist(),
        predicted_ratio=ratio, predicted_gradient=(ratio*exact).tolist(),
        online_gradient=np.asarray(actual).tolist(), bptt_gradient=np.asarray(reference).tolist(),
        finite_difference=float((plus-minus)/2e-5), exact_projection=float(exact @ np.asarray(direction)))


def validate(row):
    """Require agreement with independent exact and approximate trace formulas.

    Parameters
    ----------
    row : dict
        Result from :func:`measure`.
    """
    np.testing.assert_allclose(row['online_gradient'], row['predicted_gradient'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(row['bptt_gradient'], row['exact_gradient'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(row['finite_difference'], row['exact_projection'], rtol=1e-9, atol=1e-10)
