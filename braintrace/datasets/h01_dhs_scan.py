# DHS update equations adapted from BrainCell quad/_staggered.py.
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental DHS kernels with a fixed-size compiled loop body.

Topology preparation uses NumPy. Numerical updates retain JAX arrays and
BrainUnit quantities. These kernels do not replace the default integrator.
"""
import brainstate
import brainunit as u
import jax.lax
import jax.numpy as jnp
import numpy as np
from braincell._misc import is_traced_value
from braincell.quad import _staggered as original
from braincell.quad import register_integrator


def _prepare_levels(edges, offsets, sentinel):
    """Pad static edge levels with the unused sentinel row."""
    counts = np.diff(offsets)
    width = int(counts.max(initial=0))
    children = np.full((len(counts), width), sentinel, dtype=np.int32)
    parents = children.copy()
    valid = np.zeros(children.shape, dtype=bool)
    for i, count in enumerate(counts):
        children[i, :count] = edges[offsets[i]:offsets[i+1], 0]
        parents[i, :count] = edges[offsets[i]:offsets[i+1], 1]
        valid[i, :count] = True
    return jnp.asarray(children), jnp.asarray(parents), jnp.asarray(valid)


def _triang(diags, solves, lowers, uppers, levels):
    """Eliminate complete tree levels in the original order."""
    if levels[0].shape[0] == 0:
        return diags, solves

    d_unit = u.get_unit(diags)
    s_unit = u.get_unit(solves)
    d_raw = u.get_mantissa(diags)
    s_raw = u.get_mantissa(solves)
    l_raw = u.get_mantissa(lowers)
    u_raw = u.get_mantissa(uppers)
    children, parents, valid = levels
    u_c = u_raw[children]
    l_c = l_raw[children]

    is_1d = (d_raw.ndim == 1) or (d_raw.shape[0] == 1)
    if is_1d:
        d = d_raw.reshape(-1)
        s = s_raw.reshape(-1)

        def level_step_1d(carry, indices):
            d, s = carry
            c, p, v, uc, lc = indices
            multiplier = uc / d[c]
            delta_d = jnp.where(v, -lc * multiplier, 0.0)
            delta_s = jnp.where(v, -s[c] * multiplier, 0.0)
            d = d.at[p].add(delta_d)
            s = s.at[p].add(delta_s)
            return (d, s), None

        (d_out, s_out), _ = jax.lax.scan(level_step_1d, (d, s), (children, parents, valid, u_c, l_c))
        d_out = d_out.reshape(d_raw.shape)
        s_out = s_out.reshape(s_raw.shape)
    else:
        def level_step(carry, indices):
            d, s = carry
            c, p, v, uc, lc = indices
            multiplier = uc / d[:, c]
            delta_d = jnp.where(v, -lc * multiplier, 0.0)
            delta_s = jnp.where(v, -s[:, c] * multiplier, 0.0)
            d = d.at[:, p].add(delta_d)
            s = s.at[:, p].add(delta_s)
            return (d, s), None

        (d_out, s_out), _ = jax.lax.scan(level_step, (d_raw, s_raw), (children, parents, valid, u_c, l_c))

    res_d = u.Quantity(d_out, d_unit) if d_unit != u.UNITLESS else d_out
    res_s = u.Quantity(s_out, s_unit) if s_unit != u.UNITLESS else s_out
    return res_d, res_s


def _backsub(diags, solves, lowers, indices):
    """Apply the original recursive-doubling jumps in a compiled scan."""
    d_unit = u.get_unit(diags)
    s_unit = u.get_unit(solves)
    d_raw = u.get_mantissa(diags)
    s_raw = u.get_mantissa(solves)
    l_raw = u.get_mantissa(lowers)
    l_raw = l_raw.at[0].set(0.0)
    lower_effect = -l_raw / d_raw
    solve_effect = s_raw / d_raw

    def jump_step(carry, parents):
        lower, solution = carry
        return (lower * lower[:, parents], solution + lower * solution[:, parents]), None

    (_, result), _ = jax.lax.scan(jump_step, (lower_effect, solve_effect), indices)
    res_v = u.Quantity(result, s_unit / d_unit) if (s_unit / d_unit) != u.UNITLESS else result
    return res_v


def _voltage_step(target, t, dt, *args):
    """Use the installed voltage assembly with the two compiled kernels."""
    runtime = getattr(target, "_runtime", None)
    pack = getattr(runtime, "h01_dhs_pack", None)
    if pack is None:
        source = original._get_dhs_static_source(target, node_tree=target.node_tree,
            scheduling=target.node_scheduling(algorithm="dhs"))
        cache = original._get_dhs_static_cache(target, source)
        levels = _prepare_levels(source.edges_np, source.level_offsets_np, source.n_point)
        pack = (source, cache, levels)
        if runtime is not None and not is_traced_value(cache.diag_ms_inv):
            runtime.h01_dhs_pack = pack
    source, cache, levels = pack
    linear, const = original._linear_and_const_term(target, target.V.value, *args)
    numeric = original._build_dhs_numeric_state(target.V.value, linear, const,
        dt=dt, static_source=source, static_cache=cache,
        edge_point_current=original._edge_point_current(target, t=t, static_source=source))
    d, s = _triang(numeric.diags, numeric.solves, numeric.lowers, numeric.uppers, levels)
    solution = _backsub(d, s, numeric.lowers, source.backsub_indices_np)
    target.V.value = original._restore_midpoint_voltage(solution,
        dynamic_rows=source.dynamic_rows_np, target_shape=target.V.value.shape)


@register_integrator("h01_staggered_scan", category="staggered",
    description="Experimental DHS scan kernels; original staggered channel order.")
def _staggered_scan_step(target, *args):
    """Advance voltage, then channels, using the installed staggered order."""
    if not isinstance(target, original.DiffEqModule):
        raise TypeError("The H01 scan integrator requires a DiffEqModule.")
    t, dt = brainstate.environ.get("t", 0.), brainstate.environ.get("dt")
    if hasattr(target, "cache_ion_total_currents"):
        target.cache_ion_total_currents(target.V.value)
    _voltage_step(target, t, dt, *args)
    point_v = target._cv_to_point_unchecked(target.V.value) if hasattr(target, "_cv_to_point_unchecked") else target._cv_to_point(target.V.value)
    if target.ion_channel_update_order == "family":
        target._integrate_runtime_synapse_dynamics(point_v)
        target._update_ion_channel_families(point_v)
    elif target.ion_channel_update_order == "integration":
        target._update_ion_channels_by_integration(point_v)
    else:
        raise ValueError("ion_channel_update_order must be 'family' or 'integration'.")
