"""Matrix-free derivatives and contractions for sparse output-side factors."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np


def _zeros(layout, dtype):
    return tuple(jnp.zeros(shape + (len(row),), dtype=dtype)
                 for shape, row in zip(layout.shapes, layout.outputs))


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
    if not layout.color_count:
        return _zeros(layout, output.dtype)

    if layout.color_count == 1:
        out_seed = jnp.ones_like(output)
        _, tan = jax.jvp(tail, (output,), (out_seed,))
        result = []
        for t, shape, row in zip(tan, layout.shapes, layout.outputs):
            if not row:
                result.append(jnp.zeros(shape + (0,), dtype=output.dtype))
            else:
                result.append(t[..., None])
        return tuple(result)

    colors_np = np.asarray(layout.colors, dtype=np.int32)
    out_seeds = jnp.asarray(colors_np[None, :] == np.arange(layout.color_count)[:, None], dtype=output.dtype)

    def _jvp_tail(out_s):
        _, tan = jax.jvp(tail, (output,), (out_s,))
        return tan

    tangents = jax.vmap(_jvp_tail, in_axes=0)(out_seeds)

    result = []
    for i, (shape, row) in enumerate(zip(layout.shapes, layout.outputs)):
        if not row:
            result.append(jnp.zeros(shape + (0,), dtype=output.dtype))
            continue
        row_colors = colors_np[np.asarray(row, dtype=np.int32)]
        if len(row_colors) == layout.color_count and np.array_equal(row_colors, np.arange(layout.color_count)):
            cols = tangents[i]
        else:
            cols = tangents[i][jnp.asarray(row_colors, dtype=jnp.int32)]
        result.append(jnp.moveaxis(cols, 0, -1))
    return tuple(result)


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
    if not layout.color_count:
        return tuple(jnp.zeros_like(factor) for factor in factors)

    if layout.color_count == 1:
        hid_seeds = tuple(
            jnp.zeros(shape, dtype=factor.dtype) if not row else factor[..., 0]
            for factor, shape, row in zip(factors, layout.shapes, layout.outputs)
        )
        _, tan = jax.jvp(transition, (state,), (hid_seeds,))
        result = []
        for t, shape, row in zip(tan, layout.shapes, layout.outputs):
            if not row:
                result.append(jnp.zeros(shape + (0,), dtype=t.dtype))
            else:
                result.append(t[..., None])
        return tuple(result)

    colors_np = np.asarray(layout.colors, dtype=np.int32)
    hidden_seeds = []
    for factor, shape, row in zip(factors, layout.shapes, layout.outputs):
        if not row:
            hidden_seeds.append(jnp.zeros((layout.color_count,) + shape, dtype=factor.dtype))
        else:
            row_colors = colors_np[np.asarray(row, dtype=np.int32)]
            moved = jnp.moveaxis(factor, -1, 0)
            if len(row_colors) == layout.color_count and np.array_equal(row_colors, np.arange(layout.color_count)):
                hid_s = moved
            else:
                hid_s = jnp.zeros((layout.color_count,) + shape, dtype=factor.dtype).at[
                    jnp.asarray(row_colors, dtype=jnp.int32)
                ].set(moved)
            hidden_seeds.append(hid_s)

    def _jvp_trans(hid_s):
        _, tan = jax.jvp(transition, (state,), (hid_s,))
        return tan

    tangents = jax.vmap(_jvp_trans, in_axes=0)(tuple(hidden_seeds))

    result = []
    for i, (shape, row) in enumerate(zip(layout.shapes, layout.outputs)):
        if not row:
            result.append(jnp.zeros_like(factors[i]))
            continue
        row_colors = colors_np[np.asarray(row, dtype=np.int32)]
        if len(row_colors) == layout.color_count and np.array_equal(row_colors, np.arange(layout.color_count)):
            cols = tangents[i]
        else:
            cols = tangents[i][jnp.asarray(row_colors, dtype=jnp.int32)]
        result.append(jnp.moveaxis(cols, 0, -1))
    return tuple(result)


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
        if not row:
            continue
        if len(row) == 1:
            value = jnp.sum(factor[..., 0] * cotangent)
            result = result.at[row[0]].add(value)
        else:
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
    if not layout.color_count:
        return _zeros(layout, output.dtype)

    if layout.color_count == 1:
        out_seed = (1.0 - decay) * jnp.ones_like(output)
        hid_seeds = tuple(
            jnp.zeros(shape, dtype=output.dtype) if not row else decay * factor[..., 0]
            for factor, shape, row in zip(factors, layout.shapes, layout.outputs)
        )
        _, tan = jax.jvp(transition, (output, state), (out_seed, hid_seeds))
        result = []
        for t, shape, row in zip(tan, layout.shapes, layout.outputs):
            if not row:
                result.append(jnp.zeros(shape + (0,), dtype=output.dtype))
            else:
                result.append(t[..., None])
        return tuple(result)

    colors_np = np.asarray(layout.colors, dtype=np.int32)
    out_seeds = (1.0 - decay) * jnp.asarray(
        colors_np[None, :] == np.arange(layout.color_count)[:, None], dtype=output.dtype
    )

    hidden_seeds = []
    for factor, shape, row in zip(factors, layout.shapes, layout.outputs):
        if not row:
            hidden_seeds.append(jnp.zeros((layout.color_count,) + shape, dtype=output.dtype))
        else:
            row_colors = colors_np[np.asarray(row, dtype=np.int32)]
            moved = decay * jnp.moveaxis(factor, -1, 0)
            if len(row_colors) == layout.color_count and np.array_equal(row_colors, np.arange(layout.color_count)):
                hid_s = moved
            else:
                hid_s = jnp.zeros((layout.color_count,) + shape, dtype=factor.dtype).at[
                    jnp.asarray(row_colors, dtype=jnp.int32)
                ].set(moved)
            hidden_seeds.append(hid_s)

    def _jvp_step(out_s, hid_s):
        _, tan = jax.jvp(transition, (output, state), (out_s, hid_s))
        return tan

    tangents = jax.vmap(_jvp_step, in_axes=(0, 0))(out_seeds, tuple(hidden_seeds))

    result = []
    for i, (shape, row) in enumerate(zip(layout.shapes, layout.outputs)):
        if not row:
            result.append(jnp.zeros(shape + (0,), dtype=output.dtype))
            continue
        row_colors = colors_np[np.asarray(row, dtype=np.int32)]
        if len(row_colors) == layout.color_count and np.array_equal(row_colors, np.arange(layout.color_count)):
            cols = tangents[i]
        else:
            cols = tangents[i][jnp.asarray(row_colors, dtype=jnp.int32)]
        result.append(jnp.moveaxis(cols, 0, -1))
    return tuple(result)
