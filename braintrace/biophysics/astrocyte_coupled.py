"""Matrix-free coupled backward Euler for calcium reactions and diffusion."""

import brainstate
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from .astrocyte import calcium_derivative


def _diffusion(graph, values):
    concentration = values.reshape(-1)
    left, right = graph.edges[:, 0], graph.edges[:, 1]
    flux = graph.conductance*(concentration[right]-concentration[left])
    rate = jnp.zeros_like(concentration).at[left].add(flux).at[right].add(-flux)
    return (rate/graph.volumes).reshape(values.shape)


def coupled_calcium_step(state, parameters, graphs, *, dt_ms, surface_to_volume,
                         glutamate_mm, influx_mm_ms, clamped_ip3_mm=None):
    """Solve reactions and all three transported species in the same time step.

    Parameters
    ----------
    state : array
        Segment-major physical state of shape (n_segments, 25).
    parameters : CalciumParameters
        Fixed biological rates.
    graphs : tuple of DiffusionGraph
        Calcium, free-mobile-buffer and bound-mobile-buffer spatial geometry.
        Closed boundaries are required for this intracellular operator.
    dt_ms : float
        Positive physical time step.
    surface_to_volume, glutamate_mm, influx_mm_ms, clamped_ip3_mm : arrays
        Per-segment reaction inputs, as in AstrocyteCalcium.update.

    Returns
    -------
    tuple
        Updated physical state, validity and maximum nonlinear residual.

    Notes
    -----
    Newton-Krylov uses Jacobian-vector products and independent 25-by-25
    reaction-block preconditioners. It never forms a compartment-squared
    Jacobian. Implicit differentiation solves the same coupled linear system.
    """
    if len(graphs) != 3 or any(not graph.closed for graph in graphs):
        raise ValueError('Calcium diffusion requires three closed intracellular graphs')
    if clamped_ip3_mm is not None:
        state = state.at[:, 24].set(clamped_ip3_mm)
    clamp_axis = None if clamped_ip3_mm is None else 0
    def local(value, surface, glutamate, influx, clamp):
        return calcium_derivative(value, parameters, surface_to_volume=surface,
            glutamate_mm=glutamate, influx_mm_ms=influx, clamped_ip3_mm=clamp)[0]
    axes = (0, 0, 0, 0, clamp_axis)
    args = (surface_to_volume, glutamate_mm, influx_mm_ms, clamped_ip3_mm)
    def equation(value):
        rate = jax.vmap(local, in_axes=axes)(value, *args)
        for graph, start in zip(graphs, (0, 12, 16)):
            rate = rate.at[:, start:start+4].add(_diffusion(graph, value[:, start:start+4]))
        return value-state-dt_ms*rate

    degree = jnp.zeros_like(state)
    for graph, start in zip(graphs, (0, 12, 16)):
        left, right = graph.edges[:, 0], graph.edges[:, 1]
        diagonal = jnp.zeros_like(graph.volumes).at[left].add(graph.conductance).at[right].add(graph.conductance)
        degree = degree.at[:, start:start+4].set((diagonal/graph.volumes).reshape(-1, 4))

    def linear_solve(operator, rhs, preconditioner=None):
        def checked_solve(op, value, precondition=None):
            result, _ = gmres(op, value, tol=1e-11, atol=0., restart=20, maxiter=20,
                              M=precondition, solve_method='incremental')
            scale = jnp.maximum(jnp.linalg.norm(value), 1e-12)
            valid = jnp.linalg.norm(op(result)-value) <= 1e-9*scale
            return jnp.where(valid, result, jnp.nan)
        # Keep numerical convergence checks inside the solve primitive. A
        # value-dependent select outside it is not a linear tangent map and
        # cannot be transposed by custom_root's reverse-mode rule.
        return jax.lax.custom_linear_solve(operator, rhs,
            solve=lambda op, value: checked_solve(op, value, preconditioner),
            transpose_solve=lambda op, value: checked_solve(op, value))

    def solve(f, initial):
        def unfinished(carry):
            _, iteration, error = carry
            return (iteration < 12) & (error > 1e-13)
        def step(carry):
            value, iteration, _ = carry
            jacobian = jax.vmap(jax.jacfwd(local), in_axes=axes)(value, *args)
            blocks = jnp.eye(25)[None, :, :]-dt_ms*jacobian
            indices = jnp.arange(25)
            blocks = blocks.at[:, indices, indices].add(dt_ms*degree)
            def preconditioner(rhs):
                return jax.vmap(jnp.linalg.solve)(blocks, rhs)
            operator = lambda vector: jax.jvp(f, (value,), (vector,))[1]
            updated = value-linear_solve(operator, f(value), preconditioner)
            return updated, iteration+1, jnp.max(jnp.abs(f(updated)))
        return brainstate.transform.while_loop(unfinished, step,
            (initial, jnp.asarray(0, dtype=jnp.int32), jnp.max(jnp.abs(f(initial)))))[0]

    updated = jax.lax.custom_root(equation, state, solve, lambda operator, rhs: linear_solve(operator, rhs))
    residual = jnp.max(jnp.abs(equation(updated)))
    valid = (jnp.all(jnp.isfinite(updated)) & jnp.all(updated >= 0) &
             jnp.all(updated[:, 20:24] <= 1) & (residual < 1e-10) & (dt_ms > 0))
    return updated, valid, residual
