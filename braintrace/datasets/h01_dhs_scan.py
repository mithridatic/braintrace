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
import numpy as np
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
    return children, parents, valid


def _triang(diags, solves, lowers, uppers, levels):
    """Eliminate complete tree levels in the original order."""
    if levels[0].shape[0] == 0:
        return diags, solves

    def level_step(carry, indices):
        d, s = carry
        children, parents, valid = indices
        multiplier = uppers[children] / d[:, children]
        delta_d = -lowers[children] * multiplier
        delta_s = -s[:, children] * multiplier
        d = d.at[:, parents].add(u.math.where(valid, delta_d, u.math.zeros_like(delta_d)))
        s = s.at[:, parents].add(u.math.where(valid, delta_s, u.math.zeros_like(delta_s)))
        return (d, s), None

    result, _ = brainstate.transform.scan(level_step, (diags, solves), levels)
    return result


def _backsub(diags, solves, lowers, indices):
    """Apply the original recursive-doubling jumps in a compiled scan."""
    original._check_comp_backsub(diags, solves, lowers, indices)
    zero = 0.*u.UNITLESS if isinstance(lowers, u.Quantity) else 0.
    lowers = lowers.at[0].set(zero)
    lower_effect, solve_effect = -lowers/diags, solves/diags

    def jump_step(carry, parents):
        lower, solution = carry
        return (lower*lower[:, parents], solution+lower*solution[:, parents]), None

    (_, result), _ = brainstate.transform.scan(jump_step, (lower_effect, solve_effect), indices)
    return result


def _voltage_step(target, t, dt, *args):
    """Use the installed voltage assembly with the two compiled kernels."""
    source = original._get_dhs_static_source(target, node_tree=target.node_tree,
        scheduling=target.node_scheduling(algorithm="dhs"))
    cache = original._get_dhs_static_cache(target, source)
    stored = getattr(target._runtime, "h01_scan_levels", None)
    if stored is None or stored[0] is not source:
        stored = (source, _prepare_levels(source.edges_np, source.level_offsets_np, source.n_point))
        target._runtime.h01_scan_levels = stored
    linear, const = original._linear_and_const_term(target, target.V.value, *args)
    numeric = original._build_dhs_numeric_state(target.V.value, linear, const,
        dt=dt, static_source=source, static_cache=cache,
        edge_point_current=original._edge_point_current(target, t=t, static_source=source))
    d, s = _triang(numeric.diags, numeric.solves, numeric.lowers, numeric.uppers, stored[1])
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
