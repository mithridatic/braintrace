"""Matrix-free derivatives and contractions for sparse output-side factors."""

import brainstate
import jax
import jax.numpy as jnp


def _zeros(layout, dtype):
    return tuple(jnp.zeros(shape + (len(row),), dtype=dtype)
                 for shape, row in zip(layout.shapes, layout.outputs))


def _selectors(layout, color):
    colors = jnp.asarray(layout.colors, dtype=jnp.int32)
    return tuple(colors[jnp.asarray(row, dtype=jnp.int32)] == color for row in layout.outputs)


def _accumulate(factors, tangents, selectors):
    return tuple(factor + tangent[..., None] * select
                 for factor, tangent, select in zip(factors, tangents, selectors, strict=True))


def instant_factors(tail, output, layout):
    """Differentiate an ETP output's tail into sparse heterogeneous state blocks.

    Parameters
    ----------
    tail : callable
        Pure function from a flattened ETP output to a tuple of state arrays.
    output : array
        Current flattened ETP output.
    layout : SparseInfluence
        Conservative, temporally closed dependency layout.

    Returns
    -------
    tuple of arrays
        One factor array of shape ``(*state_shape, support_width)`` per block.

    Notes
    -----
    Simultaneous output directions may share a color only when their reachable
    state blocks are disjoint. The scan retains sparse factors and one JVP's
    working state, rather than stacking full-state derivatives for every color.
    """
    if output.shape != (len(layout.colors),):
        raise ValueError('ETP output shape does not match sparse layout')
    factors = _zeros(layout, output.dtype)
    if not layout.color_count:
        return factors
    colors = jnp.asarray(layout.colors, dtype=jnp.int32)

    def direction(carry, color):
        seed = (colors == color).astype(output.dtype)
        _, tangents = jax.jvp(tail, (output,), (seed,))
        return _accumulate(carry, tangents, _selectors(layout, color)), None

    return brainstate.transform.scan(direction, factors, jnp.arange(layout.color_count))[0]


def propagate_factors(transition, state, factors, layout):
    """Propagate sparse factors through the complete recurrent state transition.

    Parameters
    ----------
    transition : callable
        Pure tuple-of-state-arrays transition with current inputs held fixed.
    state : tuple of arrays
        State at the start of the transition.
    factors : tuple of arrays
        Sparse output factors at that same boundary.
    layout : SparseInfluence
        Support closed over all transition dependencies.

    Returns
    -------
    tuple of arrays
        The sparse representation of the state Jacobian applied to the factors.

    Notes
    -----
    This computes full JVPs through cable and cross-state operations. It does not
    replace them by per-position diagonal Jacobians. Correct sparsity depends on
    the compiler proving conservative instantaneous and temporal support.
    """
    if len(state) != len(layout.shapes) or len(factors) != len(state):
        raise ValueError('State and factor blocks must match sparse layout')
    for value, factor, shape, row in zip(state, factors, layout.shapes, layout.outputs):
        if value.shape != shape or factor.shape != shape + (len(row),):
            raise ValueError('State or factor shape does not match sparse layout')
    result = tuple(jnp.zeros_like(factor) for factor in factors)
    if not layout.color_count:
        return result

    def direction(carry, color):
        selectors = _selectors(layout, color)
        seed = tuple(jnp.sum(factor * select, axis=-1)
                     for factor, select in zip(factors, selectors))
        _, tangents = jax.jvp(transition, (state,), (seed,))
        return _accumulate(carry, tangents, selectors), None

    return brainstate.transform.scan(direction, result, jnp.arange(layout.color_count))[0]


def contract_factors(factors, cotangents, layout):
    """Contract hidden learning signals back to the original ETP output indices.

    Parameters
    ----------
    factors : tuple of arrays
        Sparse output-side factors.
    cotangents : tuple of arrays
        Learning signal for each corresponding state block.
    layout : SparseInfluence
        Sparse support and original flattened ETP output indices.

    Returns
    -------
    array
        Output cotangent for the registered ETP parameter VJP rule.
    """
    if len(factors) != len(layout.shapes) or len(cotangents) != len(factors):
        raise ValueError('Factor and cotangent blocks must match sparse layout')
    dtype = jnp.result_type(*[f.dtype for f in factors]) if factors else jnp.float32
    result = jnp.zeros(len(layout.colors), dtype=dtype)
    for factor, cotangent, shape, row in zip(factors, cotangents, layout.shapes, layout.outputs):
        if factor.shape != shape + (len(row),) or cotangent.shape != shape:
            raise ValueError('Factor or cotangent shape does not match sparse layout')
        value = jnp.sum(factor * cotangent[..., None], axis=tuple(range(len(shape))))
        result = result.at[jnp.asarray(row, dtype=jnp.int32)].add(value)
    return result


def advance_factors(transition, output, state, factors, layout, decay):
    """Compute propagated and injected factors in one joint state/output JVP.

    Parameters
    ----------
    transition : callable
        Pure function from output and state to next state.
    output : array
        Flattened registered ETP outputs.
    state, factors : tuples
        Current physical state and sparse output factors.
    layout : SparseInfluence
        Conservative closed support layout.
    decay : float
        Output-factor smoothing coefficient.

    Returns
    -------
    tuple
        ``decay * J_state @ factors + (1-decay) * J_output`` in sparse form.
    """
    result = _zeros(layout, output.dtype)
    if not layout.color_count:
        return result
    colors = jnp.asarray(layout.colors, dtype=jnp.int32)
    def direction(carry, color):
        selectors = _selectors(layout, color)
        hidden_seed = tuple(decay*jnp.sum(factor*select, axis=-1)
                            for factor, select in zip(factors, selectors, strict=True))
        output_seed = (1-decay)*(colors == color).astype(output.dtype)
        _, tangent = jax.jvp(transition, (output, state), (output_seed, hidden_seed))
        return _accumulate(carry, tangent, selectors), None
    return brainstate.transform.scan(direction, result, jnp.arange(layout.color_count))[0]
