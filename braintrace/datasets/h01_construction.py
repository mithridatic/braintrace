"""Bound branch-index work and eliminate dense axial solver overhead during H01 construction."""

import bisect
import sys
from contextlib import contextmanager
from weakref import WeakValueDictionary

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np
import saiunit._backend
import saiunit._base_dimension
import saiunit._base_unit
import saiunit._base_quantity

from braincell._base import IonChannel
from braincell._misc import is_traced_value
from braincell._compute import runtime as _runtime_module
from braincell._compute.runtime import (
    CellRuntimeState,
    _is_root_level_runtime_node,
    clone_morpho,
    cv_value_vector,
)
from braincell._multi_compartment import bridge
from braincell._multi_compartment.cell import AxialOperatorCache
from braincell.quad import _staggered as _st
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqState

import braincell._discretization.geometry as _geo_mod
import braincell._discretization.base as _base_mod
import braincell._discretization.mechanism as _mech_mod
import braincell._discretization.node_build as _node_build_mod
import braincell._discretization.policy as _policy_mod
import braincell.filter.helper as _filter_helper


# 1. Avoid dynamic import of unimported heavy backends (like PyTorch) during scalar checks
_orig_try_import = saiunit._backend._try_import
def _fast_try_import(module_name: str):
    root_mod = module_name.split(".")[0]
    if root_mod in ("torch", "cupy", "dask", "ndonnx") and root_mod not in sys.modules:
        return None
    return _orig_try_import(module_name)

saiunit._backend._try_import = _fast_try_import

# 2. Fast Unit and Dimension comparisons
def _fast_dim_eq(self, other):
    if self is other:
        return True
    if not isinstance(other, saiunit._base_dimension.Dimension):
        return NotImplemented
    if self._dims is other._dims:
        return True
    return np.array_equal(self._dims, other._dims)

saiunit._base_dimension.Dimension.__eq__ = _fast_dim_eq

_orig_has_same_dim = saiunit._base_unit.Unit.has_same_dim
def _fast_has_same_dim(self, other):
    if self is other:
        return True
    if self._dim is not None and hasattr(other, "_dim") and other._dim is not None:
        if self._dim is other._dim:
            return True
    return _orig_has_same_dim(self, other)

saiunit._base_unit.Unit.has_same_dim = _fast_has_same_dim

def _fast_has_same_mag(self, other):
    if self is other:
        return True
    if not isinstance(other, saiunit._base_unit.Unit):
        return False
    return self.scale == other.scale and self.base == other.base and self.factor == other.factor

saiunit._base_unit.Unit.has_same_magnitude = _fast_has_same_mag

_orig_to_decimal = saiunit._base_quantity.Quantity.to_decimal
def _fast_to_decimal(self, unit=saiunit._base_unit.UNITLESS):
    if self.unit is unit:
        return self.mantissa
    return _orig_to_decimal(self, unit)

saiunit._base_quantity.Quantity.to_decimal = _fast_to_decimal

_orig_in_unit = saiunit._base_quantity.Quantity.in_unit
def _fast_in_unit(self, other=saiunit._base_unit.UNITLESS, err_msg=None):
    if self.unit is other or self.unit == other:
        return self
    return _orig_in_unit(self, other, err_msg=err_msg)

saiunit._base_quantity.Quantity.in_unit = _fast_in_unit

_orig_bin_op = saiunit._base_quantity.Quantity._binary_operation
def _fast_binary_operation(
    self,
    other,
    value_operation,
    unit_operation=lambda a, b: a,
    fail_for_mismatch=False,
    operator_str=None,
    inplace=False,
):
    if not isinstance(other, saiunit._base_quantity.Quantity):
        other = saiunit._base_quantity._to_quantity(other)

    if fail_for_mismatch:
        if self.unit is not other.unit and self.unit != other.unit:
            if (other.unit.is_unitless and not self.unit.is_unitless and saiunit._base_quantity._is_concrete_zero(other.mantissa)):
                other = saiunit._base_quantity.Quantity(other.mantissa, unit=self.unit)
            elif (self.unit.is_unitless and not other.unit.is_unitless and saiunit._base_quantity._is_concrete_zero(self.mantissa)):
                self = saiunit._base_quantity.Quantity(self.mantissa, unit=other.unit)
            else:
                other = other.in_unit(
                    self.unit,
                    err_msg=f"Cannot calculate \n{self} {operator_str} {other}, because units do not match: {self.unit} != {other.unit}"
                )
    r = saiunit._base_quantity.Quantity(
        value_operation(self.mantissa, other.mantissa),
        unit=unit_operation(self.unit, other.unit)
    )
    if inplace:
        if saiunit._base_quantity._is_tracer(self.mantissa) or isinstance(self.mantissa, saiunit._base_quantity.numbers.Number):
            return r
        self.update_mantissa(r.mantissa)
        return self
    return r

saiunit._base_quantity.Quantity._binary_operation = _fast_binary_operation

_orig_q_init = saiunit._base_quantity.Quantity.__init__
def _fast_q_init(self, mantissa, unit=saiunit._base_unit.UNITLESS, dtype=None):
    if isinstance(mantissa, np.ndarray) and isinstance(unit, saiunit._base_unit.Unit) and dtype is None:
        self._mantissa = mantissa
        self._unit = unit
        return
    _orig_q_init(self, mantissa, unit=unit, dtype=dtype)

saiunit._base_quantity.Quantity.__init__ = _fast_q_init

# 3. Fast CV geometry and frusta
def _get_branch_float_arrays(branch):
    b = getattr(branch, "_branch", branch)
    cached = getattr(b, "_h01_float_arrays", None)
    if cached is not None:
        return cached
    l_um = b.lengths.mantissa if isinstance(b.lengths, u.Quantity) else np.asarray(b.lengths, dtype=float)
    rp_um = b.radii_proximal.mantissa if isinstance(b.radii_proximal, u.Quantity) else np.asarray(b.radii_proximal, dtype=float)
    rd_um = b.radii_distal.mantissa if isinstance(b.radii_distal, u.Quantity) else np.asarray(b.radii_distal, dtype=float)
    pp_um = b.points_proximal.mantissa if isinstance(b.points_proximal, u.Quantity) else (np.asarray(b.points_proximal, dtype=float) if b.points_proximal is not None else None)
    pd_um = b.points_distal.mantissa if isinstance(b.points_distal, u.Quantity) else (np.asarray(b.points_distal, dtype=float) if b.points_distal is not None else None)
    tot_len = float(np.sum(l_um))
    seg_starts = np.concatenate(([0.0], np.cumsum(l_um)[:-1]))
    seg_ends = seg_starts + l_um
    cached = (l_um, rp_um, rd_um, pp_um, pd_um, tot_len, seg_starts, seg_ends)
    try:
        object.__setattr__(b, "_h01_float_arrays", cached)
    except (AttributeError, TypeError):
        pass
    return cached

def _fast_build_frusta(branch, *, prox: float, dist: float):
    prox_f = float(prox)
    dist_f = float(dist)
    if not (0.0 - _geo_mod.EPS_PARAM <= prox_f < dist_f - _geo_mod.EPS_PARAM and dist_f <= 1.0 + _geo_mod.EPS_PARAM):
        raise ValueError(f"CV bounds must satisfy 0 <= prox < dist <= 1, got {(prox, dist)!r}.")
    prox_f = max(0.0, min(1.0, prox_f))
    dist_f = max(0.0, min(1.0, dist_f))

    (lengths_um, radii_prox_um, radii_dist_um, points_proximal, points_distal,
     total_length_um, segment_starts_um, segment_ends_um) = _get_branch_float_arrays(branch)

    if total_length_um <= _geo_mod.EPS_LEN_UM:
        raise ValueError(f"Branch total length must be > {_geo_mod.EPS_LEN_UM} μm (got {total_length_um} μm).")

    if len(lengths_um) == 1:
        seg_length_um = float(lengths_um[0])
        r_seg_prox = float(radii_prox_um[0])
        r_seg_dist = float(radii_dist_um[0])
        r0_um = r_seg_prox + (r_seg_dist - r_seg_prox) * prox_f
        r1_um = r_seg_prox + (r_seg_dist - r_seg_prox) * dist_f
        point0 = None
        point1 = None
        if points_proximal is not None and points_distal is not None:
            p_prox = points_proximal[0]
            p_dist = points_distal[0]
            point0 = p_prox + (p_dist - p_prox) * prox_f
            point1 = p_prox + (p_dist - p_prox) * dist_f
        return (_geo_mod._Frustum(prox=prox_f, dist=dist_f, length_um=(dist_f - prox_f) * seg_length_um,
                                  r_prox_um=r0_um, r_dist_um=r1_um, point_prox_um=point0, point_dist_um=point1),)

    start_um = prox_f * total_length_um
    end_um = dist_f * total_length_um

    start_idx = max(0, int(np.searchsorted(segment_ends_um, start_um - _geo_mod.EPS_LEN_UM, side='left')))
    end_idx = min(len(lengths_um), int(np.searchsorted(segment_starts_um, end_um + _geo_mod.EPS_LEN_UM, side='right')) + 1)

    frusta = []
    for seg_idx in range(start_idx, end_idx):
        seg_length_um = float(lengths_um[seg_idx])
        seg_start_um = float(segment_starts_um[seg_idx])
        seg_end_um = float(segment_ends_um[seg_idx])
        if seg_length_um <= _geo_mod.EPS_LEN_UM:
            r_seg_prox = float(radii_prox_um[seg_idx])
            r_seg_dist = float(radii_dist_um[seg_idx])
            if np.isclose(r_seg_prox, r_seg_dist):
                continue
            if not _geo_mod._owns_zero_length_jump(position_um=seg_start_um, start_um=start_um, end_um=end_um):
                continue
            x_jump = max(prox_f, min(dist_f, seg_start_um / total_length_um))
            point_jump = points_distal[seg_idx] if points_distal is not None else None
            frusta.append(_geo_mod._Frustum(prox=float(x_jump), dist=float(x_jump), length_um=0.0, r_prox_um=r_seg_prox, r_dist_um=r_seg_dist, point_prox_um=point_jump, point_dist_um=point_jump))
            continue

        left_um = max(seg_start_um, start_um)
        right_um = min(seg_end_um, end_um)
        if right_um - left_um <= _geo_mod.EPS_LEN_UM:
            continue

        t0 = (left_um - seg_start_um) / seg_length_um
        t1 = (right_um - seg_start_um) / seg_length_um
        r_seg_prox = float(radii_prox_um[seg_idx])
        r_seg_dist = float(radii_dist_um[seg_idx])
        r0_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t0
        r1_um = r_seg_prox + (r_seg_dist - r_seg_prox) * t1

        x0 = max(prox_f, min(dist_f, left_um / total_length_um))
        x1 = max(prox_f, min(dist_f, right_um / total_length_um))
        point0 = None
        point1 = None
        if points_proximal is not None and points_distal is not None:
            p_prox = points_proximal[seg_idx]
            p_dist = points_distal[seg_idx]
            point0 = p_prox + (p_dist - p_prox) * t0
            point1 = p_prox + (p_dist - p_prox) * t1

        frusta.append(_geo_mod._Frustum(prox=float(x0), dist=float(x1), length_um=float(right_um - left_um), r_prox_um=r0_um, r_dist_um=r1_um, point_prox_um=point0, point_dist_um=point1))

    if len(frusta) == 0:
        raise ValueError(f"CV [{prox}, {dist}] produced no frusta.")
    return tuple(frusta)

_geo_mod._build_frusta = _fast_build_frusta

def _fast_bounds_from_max_len_um(branch, *, max_len_um: float, keep_odd: bool):
    b = getattr(branch, "_branch", branch)
    cached = getattr(b, "_h01_float_arrays", None)
    if cached is not None:
        branch_len_um = float(cached[5])
    else:
        branch_len_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    if branch_len_um <= max_len_um + _geo_mod.EPS_LEN_UM:
        return ((0.0, 1.0),)
    n_cv = int(np.ceil((branch_len_um / max_len_um) - _geo_mod.EPS_PARAM))
    n_cv = max(1, n_cv)
    if keep_odd and n_cv % 2 == 0:
        n_cv += 1
    return tuple(
        (float(offset) / float(n_cv), float(offset + 1) / float(n_cv))
        for offset in range(n_cv)
    )

_policy_mod._bounds_from_max_len_um = _fast_bounds_from_max_len_um

_orig_validate_morphology = _geo_mod.validate_morphology
def _fast_validate_morphology(morpho):
    if getattr(morpho, "_h01_validated", False):
        return
    for branch_id, view in enumerate(morpho.branches):
        b = getattr(view, "_branch", view)
        cached = getattr(b, "_h01_float_arrays", None)
        if cached is not None:
            tot_len = cached[5]
            if tot_len <= _geo_mod.EPS_LEN_UM:
                raise ValueError(f"Branch {branch_id} has total length <= {_geo_mod.EPS_LEN_UM} μm; morphology rejected.")
            radii_prox = cached[1]
            radii_dist = cached[2]
            if np.any(radii_prox <= 0.0) or np.any(radii_dist <= 0.0):
                raise ValueError(f"Branch {branch_id} has non-positive radii; morphology rejected.")
        else:
            _orig_validate_morphology(morpho)
            break
    try:
        morpho._h01_validated = True
    except (AttributeError, TypeError):
        pass

_geo_mod.validate_morphology = _fast_validate_morphology

_orig_build_cv_geometry = _geo_mod.build_cv_geometry
def _fast_build_cv_geometry(morpho, bounds_by_branch):
    # Branch geometries are immutable; retain their owners alongside identities
    # so recycled object addresses can never produce a false cache hit.
    owners = tuple(node.branch for node in morpho._nodes.values())
    topology = tuple((node_id, id(node.branch), node.parent_id, node.parent_x, node.child_x)
                     for node_id, node in morpho._nodes.items())
    bounds = tuple(tuple((float(lo), float(hi)) for lo, hi in branch)
                   for branch in bounds_by_branch)
    key = (topology, bounds)
    cached = getattr(morpho, "_h01_cv_geometry_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    if not getattr(morpho, "_h01_validated", False):
        _geo_mod.validate_morphology(morpho)
        _geo_mod.validate_bounds(bounds_by_branch, morpho)

    n_branches = len(morpho._nodes)
    branch_to_cv_ids_lists = []
    geos = []
    parent_by_cv = []
    children_by_cv = []

    cv_id = 0
    pi = float(np.pi)
    inv_pi_1e4 = 1e4 / pi

    for branch_id in range(n_branches):
        node = morpho._nodes[branch_id]
        branch = node.branch
        arrs = getattr(branch, "_h01_float_arrays", None)
        b_bounds = bounds_by_branch[branch_id]
        ids = []

        if arrs is not None and len(b_bounds) == 1 and b_bounds[0][0] <= 1e-9 and b_bounds[0][1] >= 1.0 - 1e-9:
            lens, r_prox_arr, r_dist_arr, pts_prox, pts_dist, tot_len, seg_starts, seg_ends = arrs
            if len(lens) == 1:
                seg_len = float(lens[0])
                rp = float(r_prox_arr[0])
                rd = float(r_dist_arr[0])
                rm = 0.5 * (rp + rd)
                slant = float(np.sqrt(seg_len**2 + (rd - rp)**2))
                area_um2 = pi * (rp + rd) * slant
                f_tot = (seg_len / (rp * rd)) * inv_pi_1e4 if (rp > 0 and rd > 0) else 0.0
                f_prox = ((0.5 * seg_len) / (rp * rm)) * inv_pi_1e4 if (rp > 0 and rm > 0) else 0.0
                f_dist = ((0.5 * seg_len) / (rm * rd)) * inv_pi_1e4 if (rm > 0 and rd > 0) else 0.0
                dm = rp + rd
                L = seg_len
            else:
                diff_r = r_dist_arr - r_prox_arr
                slants = np.sqrt(lens * lens + diff_r * diff_r)
                area_um2 = float(np.sum(pi * (r_prox_arr + r_dist_arr) * slants))
                f_tot = float(np.sum((lens / (r_prox_arr * r_dist_arr)) * inv_pi_1e4))
                half_len = 0.5 * tot_len
                mid_idx = np.searchsorted(seg_ends, half_len)
                if mid_idx >= len(lens):
                    mid_idx = len(lens) - 1
                s_start = seg_starts[mid_idx]
                s_len = lens[mid_idx]
                t_mid = (half_len - s_start) / s_len if s_len > 0 else 0.5
                rm = float(r_prox_arr[mid_idx] + (r_dist_arr[mid_idx] - r_prox_arr[mid_idx]) * t_mid)
                if mid_idx == 0:
                    L_left = half_len
                    r_p_left = float(r_prox_arr[0])
                    f_prox = float(((L_left) / (r_p_left * rm)) * inv_pi_1e4) if (r_p_left > 0 and rm > 0) else 0.0
                else:
                    f_left_prev = float(np.sum((lens[:mid_idx] / (r_prox_arr[:mid_idx] * r_dist_arr[:mid_idx])) * inv_pi_1e4))
                    L_part = half_len - s_start
                    r_p_part = float(r_prox_arr[mid_idx])
                    f_left_part = float(((L_part) / (r_p_part * rm)) * inv_pi_1e4) if (r_p_part > 0 and rm > 0) else 0.0
                    f_prox = f_left_prev + f_left_part
                f_dist = f_tot - f_prox
                rp = float(r_prox_arr[0])
                rd = float(r_dist_arr[-1])
                dm = float(np.sum((r_prox_arr + r_dist_arr) * lens) / tot_len) if tot_len > 0 else (rp + rd)
                L = tot_len

            geos.append(_geo_mod._GeoCV(
                id=cv_id,
                branch_id=branch_id,
                branch_type=branch.type,
                prox=0.0,
                dist=1.0,
                midpoint=0.5,
                parent_cv=None,
                children_cv=(),
                length_um=L,
                lateral_area_um2=area_um2,
                axial_factor_total_per_cm=f_tot,
                axial_factor_prox_per_cm=f_prox,
                axial_factor_dist_per_cm=f_dist,
                r_prox_um=rp,
                r_mid_um=rm,
                diam_arc_mean_um=dm,
                r_dist_um=rd
            ))
            parent_by_cv.append(None)
            children_by_cv.append([])
            ids.append(cv_id)
            cv_id += 1
        elif arrs is not None and len(arrs[0]) == 1:
            lens, r_prox_arr, r_dist_arr = arrs[0], arrs[1], arrs[2]
            seg_len = float(lens[0])
            r0 = float(r_prox_arr[0])
            r1 = float(r_dist_arr[0])

            for prox, dist in b_bounds:
                prox_f = float(prox)
                dist_f = float(dist)
                L = (dist_f - prox_f) * seg_len
                rp = r0 + (r1 - r0) * prox_f
                rd = r0 + (r1 - r0) * dist_f
                rm = 0.5 * (rp + rd)

                slant = float(np.sqrt(L**2 + (rd - rp)**2))
                area_um2 = pi * (rp + rd) * slant

                factor_total = (L / (rp * rd)) * inv_pi_1e4 if (rp > 0 and rd > 0) else 0.0
                factor_prox = ((0.5 * L) / (rp * rm)) * inv_pi_1e4 if (rp > 0 and rm > 0) else 0.0
                factor_dist = ((0.5 * L) / (rm * rd)) * inv_pi_1e4 if (rm > 0 and rd > 0) else 0.0

                midpoint = 0.5 * (prox_f + dist_f)

                geos.append(_geo_mod._GeoCV(
                    id=cv_id,
                    branch_id=branch_id,
                    branch_type=branch.type,
                    prox=prox_f,
                    dist=dist_f,
                    midpoint=midpoint,
                    parent_cv=None,
                    children_cv=(),
                    length_um=L,
                    lateral_area_um2=area_um2,
                    axial_factor_total_per_cm=factor_total,
                    axial_factor_prox_per_cm=factor_prox,
                    axial_factor_dist_per_cm=factor_dist,
                    r_prox_um=rp,
                    r_mid_um=rm,
                    diam_arc_mean_um=rp + rd,
                    r_dist_um=rd
                ))
                parent_by_cv.append(None)
                children_by_cv.append([])
                ids.append(cv_id)
                cv_id += 1
        else:
            for prox, dist in b_bounds:
                prox_f = float(prox)
                dist_f = float(dist)
                frusta = _geo_mod._build_frusta(branch, prox=prox_f, dist=dist_f)
                length_um = sum(p.length_um for p in frusta)
                area_um2 = _geo_mod._lateral_area_um2(frusta)
                factor_total = _geo_mod._axial_factor_per_cm(frusta)
                midpoint = 0.5 * (prox_f + dist_f)
                left, right = _geo_mod._split_frusta(frusta, x=midpoint)
                factor_prox = _geo_mod._axial_factor_per_cm(left) if left else 0.0
                factor_dist = _geo_mod._axial_factor_per_cm(right) if right else 0.0
                r_prox, r_dist = _geo_mod._boundary_radii_um(frusta)
                r_mid = _geo_mod._midpoint_radius_um(frusta)

                geos.append(
                    _geo_mod._GeoCV(
                        id=cv_id,
                        branch_id=branch_id,
                        branch_type=branch.type,
                        prox=prox_f,
                        dist=dist_f,
                        midpoint=midpoint,
                        parent_cv=None,
                        children_cv=(),
                        length_um=length_um,
                        lateral_area_um2=area_um2,
                        axial_factor_total_per_cm=factor_total,
                        axial_factor_prox_per_cm=factor_prox,
                        axial_factor_dist_per_cm=factor_dist,
                        r_prox_um=r_prox,
                        r_mid_um=r_mid,
                        diam_arc_mean_um=_geo_mod._arc_weighted_mean_diam_um(frusta),
                        r_dist_um=r_dist,
                    )
                )
                parent_by_cv.append(None)
                children_by_cv.append([])
                ids.append(cv_id)
                cv_id += 1

        branch_to_cv_ids_lists.append(tuple(ids))

    branch_to_cv_ids = tuple(branch_to_cv_ids_lists)

    for ids in branch_to_cv_ids:
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            parent_by_cv[right_id] = left_id
            children_by_cv[left_id].append(right_id)

    for edge in morpho.edges:
        parent_ids = branch_to_cv_ids[edge.parent.index]
        child_ids = branch_to_cv_ids[edge.child.index]
        if len(parent_ids) == 1:
            parent_cv = parent_ids[0]
        else:
            p_x = float(edge.parent_x)
            if p_x >= 1.0 - 1e-9:
                parent_cv = parent_ids[-1]
            elif p_x <= 1e-9:
                parent_cv = parent_ids[0]
            else:
                parent_cv = _geo_mod.locate_cv_on_branch(parent_ids, geos, x=p_x)
        child_cv = child_ids[0]
        if parent_by_cv[child_cv] is None:
            parent_by_cv[child_cv] = parent_cv
        if child_cv not in children_by_cv[parent_cv]:
            children_by_cv[parent_cv].append(child_cv)

    finalized = tuple(
        _geo_mod._GeoCV(
            id=geo.id,
            branch_id=geo.branch_id,
            branch_type=geo.branch_type,
            prox=geo.prox,
            dist=geo.dist,
            midpoint=geo.midpoint,
            parent_cv=parent_by_cv[geo.id],
            children_cv=tuple(children_by_cv[geo.id]),
            length_um=geo.length_um,
            lateral_area_um2=geo.lateral_area_um2,
            axial_factor_total_per_cm=geo.axial_factor_total_per_cm,
            axial_factor_prox_per_cm=geo.axial_factor_prox_per_cm,
            axial_factor_dist_per_cm=geo.axial_factor_dist_per_cm,
            r_prox_um=geo.r_prox_um,
            r_mid_um=geo.r_mid_um,
            diam_arc_mean_um=geo.diam_arc_mean_um,
            r_dist_um=geo.r_dist_um,
        )
        for geo in geos
    )

    res = _geo_mod.CVGeometryResult(
        geos=finalized,
        branch_to_cv_ids=branch_to_cv_ids,
    )
    morpho._h01_cv_geometry_cache = (key, res, owners)
    return res

_geo_mod.build_cv_geometry = _fast_build_cv_geometry
_base_mod.build_cv_geometry = _fast_build_cv_geometry

def _fast_build_cv_mechanisms(morpho, geometry, *, paint_rules=(), place_rules=()):
    geos = geometry.geos
    buckets = [_mech_mod._init_bucket() for _ in geos]
    cache = _mech_mod._RegionCache(morpho)
    branch_to_cv_ids = geometry.branch_to_cv_ids

    frusta_cache = {}

    def _cached_frusta(branch, *, prox, dist):
        key = (id(branch), round(float(prox), 9), round(float(dist), 9))
        cached = frusta_cache.get(key)
        if cached is None:
            cached = _geo_mod._build_frusta(branch, prox=prox, dist=dist)
            frusta_cache[key] = cached
        return cached

    for rule in paint_rules:
        intervals_by_branch = cache.intervals(rule.region)
        mechanism = rule.mechanism
        is_cable = isinstance(mechanism, _mech_mod.CableProperty)

        for branch_id, intervals in intervals_by_branch.items():
            if branch_id >= len(branch_to_cv_ids):
                continue
            cv_ids = branch_to_cv_ids[branch_id]
            if not cv_ids or not intervals:
                continue

            if len(intervals) == 1 and intervals[0][0] <= _mech_mod.EPS_PARAM and intervals[0][1] >= 1.0 - _mech_mod.EPS_PARAM:
                if is_cable:
                    for cv_id in cv_ids:
                        buckets[cv_id].cable = mechanism
                else:
                    for cv_id in cv_ids:
                        _mech_mod._apply_density(buckets[cv_id], mechanism, region_key=rule.region, fraction=1.0)
                continue

            for cv_id in cv_ids:
                geo_cv = geos[cv_id]
                bucket = buckets[cv_id]

                if is_cable:
                    if not _mech_mod._interval_contains(intervals, geo_cv.midpoint, epsilon=_mech_mod.EPS_PARAM):
                        continue
                    bucket.cable = mechanism
                    continue

                fraction = _mech_mod._coverage_fraction(morpho, geo_cv, intervals, frusta_builder=_cached_frusta)
                if fraction <= _mech_mod.EPS_PARAM:
                    continue
                _mech_mod._apply_density(bucket, mechanism, region_key=rule.region, fraction=fraction)

    seen_names = set()
    for rule in place_rules:
        for branch_id, x, display_name in cache.points(rule.locset):
            ids = geometry.cv_ids(branch_id)
            if not ids:
                continue
            cv_id = geometry.locate_cv(branch_id=branch_id, x=x)
            geo_cv = geos[cv_id]
            position = _mech_mod._position_for_geo(geo_cv, x=float(x))
            for mechanism in rule.mechanisms:
                _mech_mod._apply_place(
                    buckets[cv_id],
                    mechanism,
                    display_name=display_name,
                    seen_names=seen_names,
                    position=position,
                )

    return buckets

_mech_mod.build_cv_mechanisms = _fast_build_cv_mechanisms

_POS_ORDER = _node_build_mod._POSITION_ORDER
_NodeRole = _node_build_mod.NodeRole
_NodeEdgeRole = _node_build_mod.NodeEdgeRole
_Node = _node_build_mod.Node
_NodeEdge = _node_build_mod.NodeEdge
_NodeTree = _node_build_mod.NodeTree

def _fast_build_node_tree(morpho, *, cvs):
    n_branches = len(morpho.branches)
    cv_ids_by_branch = [[] for _ in range(n_branches)]
    for cv in cvs:
        cv_ids_by_branch[cv.branch_id].append(cv.id)

    drafts = []
    cv_to_mid_node_id = np.full(len(cvs), -1, dtype=np.int32)
    branch_endpoint_node_id_by_x = {}
    logical_edge_roles = {}
    logical_edge_order = []

    def new_node(*, cv_id, position):
        node_id = len(drafts)
        drafts.append([(cv_id, position)])
        return node_id

    def add_node_role(node_id, *, cv_id, position):
        pair = (cv_id, position)
        r = drafts[node_id]
        if pair not in r:
            r.append(pair)

    def add_edge_role(parent_node_id, child_node_id, *, cv_id, half):
        key = (parent_node_id, child_node_id)
        entry = logical_edge_roles.get(key)
        if entry is None:
            entry = []
            logical_edge_roles[key] = entry
            logical_edge_order.append(key)
        role = (cv_id, half)
        if role not in entry:
            entry.append(role)

    root_branch_cv_ids = cv_ids_by_branch[0]
    if len(root_branch_cv_ids) == 0:
        raise ValueError("Root branch has no CVs.")
    root_node_id = new_node(cv_id=root_branch_cv_ids[0], position="prox")
    branch_endpoint_node_id_by_x[(0, 0.0)] = root_node_id

    for branch_id in range(n_branches):
        branch_cv_ids = cv_ids_by_branch[branch_id]
        node = morpho._nodes[branch_id]
        if node.parent_id is None:
            attachment_node_id = root_node_id
            attach_x = 0.0
            ordered_cv_ids = branch_cv_ids
        else:
            p_id = node.parent_id
            p_x = float(node.parent_x) if node.parent_x is not None else 1.0
            if (p_id, p_x) in branch_endpoint_node_id_by_x:
                attachment_node_id = branch_endpoint_node_id_by_x[(p_id, p_x)]
            elif p_x >= 1.0 - 1e-9 and (p_id, 1.0) in branch_endpoint_node_id_by_x:
                attachment_node_id = branch_endpoint_node_id_by_x[(p_id, 1.0)]
            elif p_x <= 1e-9 and (p_id, 0.0) in branch_endpoint_node_id_by_x:
                attachment_node_id = branch_endpoint_node_id_by_x[(p_id, 0.0)]
            else:
                attachment_node_id = _node_build_mod._resolve_attachment_node(
                    p_id, parent_x=p_x,
                    branch_endpoint_node_id_by_x=branch_endpoint_node_id_by_x,
                    cv_to_mid_node_id=cv_to_mid_node_id,
                    cv_ids_by_branch=cv_ids_by_branch,
                    cvs=cvs
                )
            c_x = float(node.child_x) if node.child_x is not None else 0.0
            attach_x = c_x
            ordered_cv_ids = branch_cv_ids if attach_x <= _node_build_mod._EPS_PARAM else tuple(reversed(branch_cv_ids))

        first_cv_id = ordered_cv_ids[0]
        add_node_role(attachment_node_id, cv_id=first_cv_id, position=_node_build_mod._entry_position_for_walk(attach_x))
        branch_endpoint_node_id_by_x[(branch_id, float(attach_x))] = attachment_node_id

        for cv_id in ordered_cv_ids:
            cv_to_mid_node_id[cv_id] = new_node(cv_id=cv_id, position="mid")

        terminal_cv_id = ordered_cv_ids[-1]
        terminal_node_id = new_node(cv_id=terminal_cv_id, position=_node_build_mod._exit_position_for_walk(attach_x))
        branch_endpoint_node_id_by_x[(branch_id, float(1.0 - attach_x))] = terminal_node_id

        n_ordered = len(ordered_cv_ids)
        for index, cv_id in enumerate(ordered_cv_ids):
            mid_id = int(cv_to_mid_node_id[cv_id])
            p_node_id = attachment_node_id if index == 0 else int(cv_to_mid_node_id[ordered_cv_ids[index - 1]])
            c_node_id = terminal_node_id if index == n_ordered - 1 else int(cv_to_mid_node_id[ordered_cv_ids[index + 1]])
            add_edge_role(p_node_id, mid_id, cv_id=cv_id, half=_node_build_mod._entry_half_for_walk(attach_x))
            add_edge_role(mid_id, c_node_id, cv_id=cv_id, half=_node_build_mod._exit_half_for_walk(attach_x))

    branch_endpoint_node_id = _node_build_mod._build_branch_endpoint_node_id(
        branch_endpoint_node_id_by_x=branch_endpoint_node_id_by_x,
        n_branches=n_branches,
    )

    n_nodes = len(drafts)
    role_to_node_id = {}
    nodes_list = []
    node_density_mech_lists = [[] for _ in range(n_nodes)]
    node_point_mech_lists = [[] for _ in range(n_nodes)]

    for cv in cvs:
        mid_id = int(cv_to_mid_node_id[cv.id])
        if cv.density_mech:
            node_density_mech_lists[mid_id].extend(cv.density_mech)

    for node_id, raw_roles in enumerate(drafts):
        if len(raw_roles) == 1:
            cid, pos = raw_roles[0]
            roles_tuple = (_NodeRole(cv_id=cid, position=pos),)
            role_to_node_id[(cid, pos)] = node_id
            kind = "mid" if pos == "mid" else "boundary"
        else:
            raw_roles.sort(key=lambda item: (item[0], _POS_ORDER[item[1]]))
            roles_tuple = tuple(_NodeRole(cv_id=cid, position=pos) for cid, pos in raw_roles)
            for cid, pos in raw_roles:
                role_to_node_id[(cid, pos)] = node_id
            kind = "mid" if all(pos == "mid" for _, pos in raw_roles) else "boundary"

        nodes_list.append(_Node(
            id=node_id,
            kind=kind,
            roles=roles_tuple,
            density_mech=tuple(node_density_mech_lists[node_id]),
            point_mech=(),
        ))

    for cv in cvs:
        if cv.point_mech_roles:
            mid_id = int(cv_to_mid_node_id[cv.id])
            for placement in cv.point_mech_roles:
                n_id = role_to_node_id.get((cv.id, placement.position), mid_id)
                node_point_mech_lists[n_id].append(placement.mechanism)

    r_prox_arr = [cv.r_axial_prox for cv in cvs]
    r_dist_arr = [cv.r_axial_dist for cv in cvs]

    edges_list = []
    for edge_id, ((p_node, c_node), cv_roles) in enumerate(
        (key, logical_edge_roles[key]) for key in logical_edge_order
    ):
        if len(cv_roles) == 1:
            c_id, half = cv_roles[0]
            r = r_prox_arr[c_id] if half == "prox" else r_dist_arr[c_id]
            edge_roles = (_NodeEdgeRole(cv_id=c_id, half=half, r_axial=r),)
        else:
            cv_roles.sort(key=lambda item: (item[0], item[1]))
            edge_roles = tuple(
                _NodeEdgeRole(
                    cv_id=c_id, half=half,
                    r_axial=(r_prox_arr[c_id] if half == "prox" else r_dist_arr[c_id]),
                )
                for c_id, half in cv_roles
            )
        edges_list.append(_NodeEdge(
            id=edge_id,
            parent_node_id=p_node,
            child_node_id=c_node,
            roles=edge_roles,
        ))

    return _NodeTree(
        nodes=tuple(nodes_list),
        edges=tuple(edges_list),
        root_node_id=root_node_id,
        cv_to_mid_node_id=cv_to_mid_node_id,
        branch_endpoint_node_id=branch_endpoint_node_id,
    )

_node_build_mod.build_node_tree_from_cvs = _fast_build_node_tree
_base_mod.build_node_tree_from_cvs = _fast_build_node_tree

_PI_FLOAT = float(np.pi)

# 4. Fast mechanism coverage fraction
def _fast_coverage_fraction(morpho, geo, intervals, *, frusta_builder=None):
    if geo.lateral_area_um2 <= _mech_mod.EPS_AREA_UM2:
        return 0.0
    overlap = 0.0
    geo_prox = geo.prox
    geo_dist = geo.dist
    eps = _mech_mod.EPS_PARAM
    branch = morpho._nodes[geo.branch_id].branch if geo.branch_id in morpho._nodes else morpho.branches[geo.branch_id]
    arrs = getattr(branch, "_h01_float_arrays", None)
    is_single_seg = arrs is not None and len(arrs[0]) == 1

    if is_single_seg:
        seg_len = float(arrs[0][0])
        r0 = float(arrs[1][0])
        r1 = float(arrs[2][0])
        dr = r1 - r0
        for left, right in intervals:
            l_f = float(left)
            r_f = float(right)
            start = max(geo_prox, l_f)
            end = min(geo_dist, r_f)
            if end - start <= eps:
                continue
            if start <= geo_prox + eps and end >= geo_dist - eps:
                overlap += geo.lateral_area_um2
            else:
                rp = r0 + dr * start
                rd = r0 + dr * end
                L = (end - start) * seg_len
                slant = np.sqrt(L * L + (rd - rp) * (rd - rp))
                overlap += _PI_FLOAT * (rp + rd) * slant
    else:
        for left, right in intervals:
            l_f = float(left)
            r_f = float(right)
            start = max(geo_prox, l_f)
            end = min(geo_dist, r_f)
            if end - start <= eps:
                continue
            if start <= geo_prox + eps and end >= geo_dist - eps:
                overlap += geo.lateral_area_um2
            else:
                build = frusta_builder if frusta_builder is not None else _fast_build_frusta
                overlap += _geo_mod._lateral_area_um2(build(branch, prox=start, dist=end))

    return max(0.0, min(1.0, overlap / geo.lateral_area_um2))

_mech_mod._coverage_fraction = _fast_coverage_fraction

# 5. Fast CV assembly
_UM_UNIT = u.um
_UM2_UNIT = u.um ** 2
_OHM_UNIT = u.ohm
_RA_CACHE = {}

def _get_ra_ohm_cm(ra):
    cached = _RA_CACHE.get(id(ra))
    if cached is not None and cached[0] is ra:
        return cached[1]
    val = float(np.asarray(ra.to_decimal(u.ohm * u.cm), dtype=float))
    # Quantity does not support weak references. Retain a bounded set of owners
    # so Python object-id reuse cannot substitute another cable's resistivity.
    if len(_RA_CACHE) >= 1024:
        _RA_CACHE.clear()
    _RA_CACHE[id(ra)] = (ra, val)
    return val

def _fast_quantity(val, unit):
    q = saiunit._base_quantity.Quantity.__new__(saiunit._base_quantity.Quantity)
    q._mantissa = val
    q._unit = unit
    return q

def _fast_assemble_cv(geo, bucket):
    cable = bucket.cable
    ra = cable.axial_resistivity
    ra_ohm_cm = _get_ra_ohm_cm(ra)
    return _base_mod.CV(
        id=geo.id,
        branch_id=geo.branch_id,
        branch_type=geo.branch_type,
        prox=geo.prox,
        dist=geo.dist,
        parent_cv=geo.parent_cv,
        children_cv=geo.children_cv,
        length=_fast_quantity(geo.length_um, _UM_UNIT),
        area=_fast_quantity(geo.lateral_area_um2, _UM2_UNIT),
        cm=cable.membrane_capacitance,
        ra=cable.axial_resistivity,
        v=cable.resting_potential,
        temp=cable.temperature,
        r_axial=_fast_quantity(ra_ohm_cm * geo.axial_factor_total_per_cm, _OHM_UNIT),
        r_axial_prox=_fast_quantity(ra_ohm_cm * geo.axial_factor_prox_per_cm, _OHM_UNIT),
        r_axial_dist=_fast_quantity(ra_ohm_cm * geo.axial_factor_dist_per_cm, _OHM_UNIT),
        radius_prox=_fast_quantity(geo.r_prox_um, _UM_UNIT),
        radius_mid=_fast_quantity(geo.r_mid_um, _UM_UNIT),
        diam_arc_mean=_fast_quantity(geo.diam_arc_mean_um, _UM_UNIT),
        radius_dist=_fast_quantity(geo.r_dist_um, _UM_UNIT),
        density_mech=tuple(bucket.density_by_key.values()),
        point_mech=tuple(bucket.points),
        point_mech_roles=tuple(bucket.point_roles),
    )

_base_mod._assemble_cv = _fast_assemble_cv

# 6. Fast bridge.quantity_vector
def _fast_quantity_vector(values, *, shape=None):
    if len(values) == 0:
        return values
    first = values[0]
    target_shape = (len(values),) if shape is None else shape
    if hasattr(first, "unit"):
        unit = first.unit
        arr = np.fromiter((item.mantissa if isinstance(item, u.Quantity) else float(item) for item in values), dtype=np.float64, count=len(values))
        return u.Quantity(arr.reshape(target_shape), unit)
    return np.asarray(values).reshape(target_shape)

bridge.quantity_vector = _fast_quantity_vector
_runtime_module.quantity_vector = _fast_quantity_vector

# 7. Fast clamp routing table
def _fast_build_clamp_routing_table(*, layouts, cvs, node_tree, n_point):
    active = set()
    for layout in layouts:
        if layout.target != "point" or layout.point_index is None:
            continue
        if layout.kind not in _runtime_module.CLAMP_KINDS:
            continue
        active.update(int(pid) for pid in layout.point_index.tolist())

    if not active:
        return None

    mid_to_cv = {int(node_tree.cv_to_mid_node_id[cv.id]): cv.id for cv in cvs}
    active_midpoints = sorted(pid for pid in active if pid in mid_to_cv)
    ids = np.asarray(active_midpoints, dtype=np.int32)
    area = np.asarray([cvs[mid_to_cv[pid]].area.mantissa * 1e-8 for pid in active_midpoints], dtype=np.float64)
    if np.any(area <= 0.0):
        bad = ids[area <= 0.0].tolist()
        raise ValueError(
            "Midpoint clamp active points must have positive membrane area; "
            f"got non-positive area at point ids {bad!r}."
        )
    boundary_ids = np.asarray(
        sorted(pid for pid in active if pid not in mid_to_cv),
        dtype=np.int32,
    )
    return _runtime_module.ClampRoutingTable(
        midpoint_ids=ids,
        midpoint_area=area,
        boundary_ids=boundary_ids,
    )

_CM2_UNIT = u.cm ** 2

_orig_from_cell = _runtime_module.CellRuntimeState.from_cell

@classmethod
def _fast_from_cell(cls, cell: "braincell.Cell") -> _runtime_module.CellRuntimeState:
    node_tree = cell.node_tree
    n_point = len(node_tree.nodes)
    n_cv = len(cell.cvs)
    cv_to_mid = np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)

    grouped = {}
    cv_to_layout_lists = [[] for _ in range(n_cv)]
    point_to_layout_lists = [[] for _ in range(n_point)]
    layout_id = 0
    pop_size = tuple(cell.pop_size)

    # Fast bulk registration by unique density_mech tuples
    density_groups = {}
    for cv in cell.cvs:
        mechs = cv.density_mech
        if mechs in density_groups:
            density_groups[mechs].append(cv.id)
        else:
            density_groups[mechs] = [cv.id]

    for mechs, cv_id_list in density_groups.items():
        cv_id_arr = np.asarray(cv_id_list, dtype=np.int32)
        pt_id_arr = cv_to_mid[cv_id_arr]
        pt_id_list = pt_id_arr.tolist()
        for mechanism in mechs:
            signature = ("density",) + _runtime_module.mechanism_signature(mechanism)
            entry = grouped.get(signature)
            if entry is None:
                entry = {
                    "id": layout_id,
                    "mechanism": mechanism,
                    "target": "density",
                    "cv_ids": set(),
                    "point_ids": set(),
                }
                grouped[signature] = entry
                layout_id += 1
            entry["cv_ids"].update(cv_id_list)
            entry["point_ids"].update(pt_id_list)

    point_mech = tuple(node.point_mech for node in node_tree.nodes)
    if any(point_mech):
        for point_id, mechanisms in enumerate(point_mech):
            source_cv_ids = _runtime_module._source_cv_ids_for_point(node_tree, point_id=int(point_id))
            for mechanism in mechanisms:
                signature = ("point",) + _runtime_module.mechanism_signature(mechanism)
                entry = grouped.get(signature)
                if entry is None:
                    entry = {
                        "id": layout_id,
                        "mechanism": mechanism,
                        "target": "point",
                        "cv_ids": set(),
                        "point_ids": set(),
                    }
                    grouped[signature] = entry
                    layout_id += 1
                entry["cv_ids"].update(source_cv_ids)
                entry["point_ids"].add(int(point_id))
    else:
        for cv in cell.cvs:
            midpoint_point_id = int(cv_to_mid[cv.id])
            for mechanism in cv.point_mech:
                signature = ("point",) + _runtime_module.mechanism_signature(mechanism)
                entry = grouped.get(signature)
                if entry is None:
                    entry = {
                        "id": layout_id,
                        "mechanism": mechanism,
                        "target": "point",
                        "cv_ids": set(),
                        "point_ids": set(),
                    }
                    grouped[signature] = entry
                    layout_id += 1
                entry["cv_ids"].add(cv.id)
                entry["point_ids"].add(midpoint_point_id)

    layouts = []
    state_shapes = {}
    state_buffers = {}
    layout_mechanisms = {}
    for entry in sorted(grouped.values(), key=lambda item: int(item["id"])):
        mechanism = entry["mechanism"]
        target = str(entry["target"])
        cv_ids = tuple(sorted(int(cv_id) for cv_id in entry["cv_ids"]))
        point_ids = np.asarray(sorted(int(point_id) for point_id in entry["point_ids"]), dtype=np.int32)
        layout = _runtime_module.choose_layout(target=target)
        if layout == "dense":
            point_mask = np.zeros(n_point, dtype=bool)
            point_mask[point_ids] = True
            point_index = point_ids
            shape = pop_size + (n_point,)
        elif layout == "sparse":
            point_mask = None
            point_index = point_ids
            shape = pop_size + (len(point_ids),)
        else:
            raise ValueError(f"Unsupported layout {layout!r}.")

        layout_spec = _runtime_module.MechanismLayout(
            id=int(entry["id"]),
            kind=_runtime_module.mechanism_kind(mechanism),
            target=target,
            layout=layout,
            point_index=point_index,
            point_mask=point_mask,
            n_active=len(point_ids),
            source_cv_ids=cv_ids,
            source_rule=None,
        )
        layouts.append(layout_spec)
        layout_mechanisms[layout_spec.id] = mechanism

        lid = layout_spec.id
        for point_id in point_ids:
            point_to_layout_lists[point_id].append(lid)
        for cv_id in cv_ids:
            cv_to_layout_lists[cv_id].append(lid)

        for var_name in _runtime_module._mechanism_var_names(mechanism):
            if isinstance(mechanism, braincell.CurrentClamp) and var_name == "delay":
                quantity = _runtime_module._allocate_current_clamp_delay_buffer(
                    mechanism=mechanism,
                    pop_size=pop_size,
                    n_active=len(point_ids),
                )
                state_buffers[(layout_spec.id, var_name)] = quantity
                state_shapes[(layout_spec.id, var_name)] = quantity.mantissa.shape
                continue
            if isinstance(mechanism, braincell.CurrentClamp) and var_name in ("durations", "amplitudes"):
                quantity, mask = _runtime_module._allocate_current_clamp_buffer(
                    mechanism=mechanism,
                    var_name=var_name,
                    pop_size=pop_size,
                    n_active=len(point_ids),
                )
                state_buffers[(layout_spec.id, var_name)] = quantity
                state_buffers[(layout_spec.id, f"_mask_{var_name}")] = mask
                state_shapes[(layout_spec.id, var_name)] = quantity.mantissa.shape
                continue
            state_shapes[(layout_spec.id, var_name)] = shape
            state_buffers[(layout_spec.id, var_name)] = _runtime_module._allocate_state_buffer(
                mechanism,
                var_name=var_name,
                shape=shape,
            )

    (
        ions,
        ion_aliases,
        ion_family_candidates,
        ion_class_candidates,
        runtime_nodes,
        bound_ion_keys,
        current_owner_keys,
    ) = _runtime_module._build_runtime_nodes(
        n_point=n_point,
        layouts=tuple(layouts),
        layout_mechanisms=layout_mechanisms,
        state_buffers=state_buffers,
        pop_size=pop_size,
    )
    _fast_attach_runtime_ion_geometry(
        ions=ions,
        cvs=cell.cvs,
        point_ids=node_tree.cv_to_mid_node_id,
        n_point=n_point,
    )

    clamp_routing_table = _fast_build_clamp_routing_table(
        layouts=tuple(layouts),
        cvs=cell.cvs,
        node_tree=node_tree,
        n_point=n_point,
    )

    cv_area_decimal = np.fromiter((cv.area.mantissa * 1e-8 for cv in cell.cvs), dtype=np.float64, count=n_cv)
    cv_area = _fast_quantity(cv_area_decimal, _CM2_UNIT)
    point_area_decimal = np.zeros(n_point, dtype=np.float64)
    point_area_decimal[node_tree.cv_to_mid_node_id] = cv_area_decimal
    point_area = _fast_quantity(point_area_decimal, _CM2_UNIT)

    return cls(
        node_tree=node_tree,
        n_point=n_point,
        n_cv=n_cv,
        layouts=tuple(layouts),
        point_to_layout_ids=tuple(tuple(ids) for ids in point_to_layout_lists),
        cv_to_layout_ids=tuple(tuple(ids) for ids in cv_to_layout_lists),
        voltage_shape=pop_size + (n_point,),
        state_shapes=state_shapes,
        state_buffers=state_buffers,
        layout_mechanisms=layout_mechanisms,
        runtime_nodes=runtime_nodes,
        ions=ions,
        ion_aliases=ion_aliases,
        ion_family_candidates=ion_family_candidates,
        ion_class_candidates=ion_class_candidates,
        bound_ion_keys=bound_ion_keys,
        current_owner_keys=current_owner_keys,
        dhs_static_source_np=None,
        dhs_static_cache=None,
        axial_operator_np=None,
        axial_operator_cache=None,
        clamp_routing_table=clamp_routing_table,
        cv_area=cv_area,
        point_area=point_area,
        pop_size=pop_size,
    )

_runtime_module.CellRuntimeState.from_cell = _fast_from_cell

# 8. Fast region interval normalization
def _fast_norm_region_intervals(intervals, *, epsilon=_filter_helper.EPSILON):
    if not intervals:
        return ()
    grouped = {}
    for branch, raw_prox, raw_dist in intervals:
        prox = float(raw_prox)
        dist = float(raw_dist)
        if prox < 0.0:
            prox = 0.0 if prox >= -epsilon else prox
        elif prox > 1.0:
            prox = 1.0 if prox <= 1.0 + epsilon else prox
        if dist < 0.0:
            dist = 0.0 if dist >= -epsilon else dist
        elif dist > 1.0:
            dist = 1.0 if dist <= 1.0 + epsilon else dist
        if prox < -epsilon or dist > 1.0 + epsilon:
            raise ValueError(f"Interval coordinates must be within [0, 1], got prox={prox!r}, dist={dist!r}.")
        if dist - prox <= epsilon:
            continue
        b = int(branch)
        if b in grouped:
            grouped[b].append((prox, dist))
        else:
            grouped[b] = [(prox, dist)]

    normalized = []
    for branch in sorted(grouped):
        ranges = grouped[branch]
        if len(ranges) > 1:
            ranges.sort()
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= current_end + epsilon:
                if end > current_end:
                    current_end = end
            else:
                if current_end - current_start > epsilon:
                    normalized.append((branch, current_start, current_end))
                current_start, current_end = start, end
        if current_end - current_start > epsilon:
            normalized.append((branch, current_start, current_end))

    return tuple(normalized)

_filter_helper.normalize_region_intervals = _fast_norm_region_intervals

def _fast_difference_region_intervals(left, right, *, epsilon=_filter_helper.EPSILON):
    if not left:
        return ()
    if not right:
        return _fast_norm_region_intervals(left, epsilon=epsilon)

    right_group = _filter_helper._group_by_branch(right, epsilon=epsilon)
    if not right_group:
        return _fast_norm_region_intervals(left, epsilon=epsilon)

    left_group = _filter_helper._group_by_branch(left, epsilon=epsilon)
    out = []
    for branch, l_ranges in left_group.items():
        if branch not in right_group:
            for start, end in l_ranges:
                out.append((branch, start, end))
        else:
            for start, end in _filter_helper._difference_branch_ranges(
                l_ranges, right_group[branch], epsilon=epsilon
            ):
                out.append((branch, start, end))
    return tuple(out)

_filter_helper.difference_region_intervals = _fast_difference_region_intervals

# 9. Fast binary doubling backsub
def _fast_build_backsub_indices(parent_lookup: np.ndarray, *, n_nodes: int) -> np.ndarray:
    parent_lookup = np.asarray(parent_lookup, dtype=np.int32)
    indices = []
    new_step = 1
    current = parent_lookup
    while new_step <= max(1, n_nodes):
        indices.append(current)
        current = current[current]
        new_step = 2 * new_step
    return np.asarray(indices, dtype=np.int32)

_st._build_backsub_indices = _fast_build_backsub_indices

# 10. Fast morphology clone
def _fast_clone_morpho(morpho: braincell.Morphology) -> braincell.Morphology:
    cloned = braincell.Morphology.__new__(braincell.Morphology)
    cloned._name_to_id = dict(morpho._name_to_id)
    cloned._type_name_counters = dict(morpho._type_name_counters)
    cloned._next_id = morpho._next_id
    cloned._root_name = morpho._root_name
    cloned._root_id = morpho._root_id
    cloned._nodes = {}
    for node_id, node in morpho._nodes.items():
        new_node = braincell.morph.MorphoBranch(
            cloned,
            node.index,
            name=node.name,
            branch=node.branch,
            parent_id=node.parent_id,
            parent_x=node.parent_x,
            child_x=node.child_x,
        )
        new_node._children = dict(node._children)
        cloned._nodes[node_id] = new_node
    sig = getattr(morpho, "_h01_geom_sig", None)
    if sig is not None:
        cloned._h01_geom_sig = sig
    cv_cache = getattr(morpho, "_h01_cv_bounds_cache", None)
    if cv_cache is not None:
        cloned._h01_cv_bounds_cache = cv_cache
    geo_cache = getattr(morpho, "_h01_cv_geometry_cache", None)
    if geo_cache is not None:
        cloned._h01_cv_geometry_cache = geo_cache
    disc_cache = getattr(morpho, "_h01_disc_cache", None)
    if disc_cache is not None:
        cloned._h01_disc_cache = disc_cache
    bmap_cache = getattr(morpho, "_cached_branch_index_map", None)
    if bmap_cache is not None:
        cloned._cached_branch_index_map = bmap_cache
    nids_cache = getattr(morpho, "_cached_ordered_node_ids", None)
    if nids_cache is not None:
        cloned._cached_ordered_node_ids = nids_cache
    return cloned

braincell.morph.morphology.clone_morpho = _fast_clone_morpho
_runtime_module.clone_morpho = _fast_clone_morpho


_cached_baseline_ions = {}


def _fast_instantiate_runtime_ion_instance(
    *, instance_name, runtime_cls, layouts, declarations, state_buffers, n_point, pop_size=()
):
    supported_params = _runtime_module._supported_ion_runtime_params(runtime_cls)
    full_size = pop_size + (n_point,)
    if runtime_cls not in _cached_baseline_ions:
        _cached_baseline_ions[runtime_cls] = runtime_cls(size=1)
    baseline_ion = _cached_baseline_ions[runtime_cls]
    full_param_values = {}
    for param_name in supported_params:
        baseline_value = _runtime_module._normalize_ion_runtime_param_value(
            runtime_cls,
            param_name,
            getattr(baseline_ion, _runtime_module._ion_runtime_attr_name(runtime_cls, param_name)),
        )
        full_param_values[param_name] = _runtime_module._ion_param_broadcast(baseline_value, shape=full_size)
    for layout, declaration in zip(layouts, declarations):
        point_index = layout.point_index
        for param_name in declaration.params.keys():
            buffer = state_buffers[(layout.id, param_name)]
            full_param_values[param_name] = _runtime_module._ion_param_scatter(
                runtime_cls=runtime_cls,
                param_name=param_name,
                target=full_param_values[param_name],
                buffer=buffer,
                point_index=point_index,
            )
    runtime_ion_instance = runtime_cls(size=full_size, name=instance_name, **full_param_values)
    _runtime_module._restore_shaped_species_initializers(runtime_ion_instance, full_param_values)
    return runtime_ion_instance


_orig_ordered_node_ids_by = braincell.Morphology._ordered_node_ids_by

def _fast_ordered_node_ids_by(self, order: str = "default"):
    cached = getattr(self, "_cached_ordered_node_ids", None)
    if cached is not None and cached[0] == (len(self._nodes), self._next_id, order):
        return cached[1]
    res = _orig_ordered_node_ids_by(self, order=order)
    self._cached_ordered_node_ids = ((len(self._nodes), self._next_id, order), res)
    return res

def _fast_branch_index_map(self, *, order: str = "default") -> dict[int, int]:
    cached = getattr(self, "_cached_branch_index_map", None)
    if cached is not None and cached[0] == (len(self._nodes), self._next_id, order):
        return cached[1]
    res = {node_id: index for index, node_id in enumerate(self._ordered_node_ids_by(order=order))}
    self._cached_branch_index_map = ((len(self._nodes), self._next_id, order), res)
    return res

def _fast_branch_index(self, node_id: int, *, order: str = "default") -> int:
    cached = getattr(self, "_cached_branch_index_map", None)
    if cached is not None and cached[0] == (len(self._nodes), self._next_id, order):
        return cached[1][node_id]
    return self._branch_index_map(order=order)[node_id]

braincell.Morphology._ordered_node_ids_by = _fast_ordered_node_ids_by
braincell.Morphology._branch_index_map = _fast_branch_index_map
braincell.Morphology._branch_index = _fast_branch_index


def _fast_attach_runtime_ion_geometry(*, ions, cvs, point_ids, n_point):
    if not ions or len(cvs) == 0:
        return
    n_cv = len(cvs)
    length = np.fromiter((cv.length.mantissa if isinstance(cv.length, u.Quantity) else float(cv.length) for cv in cvs), dtype=np.float64, count=n_cv)
    area = np.fromiter((cv.area.mantissa if isinstance(cv.area, u.Quantity) else float(cv.area) for cv in cvs), dtype=np.float64, count=n_cv)
    radius_prox = np.fromiter((cv.radius_prox.mantissa if isinstance(cv.radius_prox, u.Quantity) else float(cv.radius_prox) for cv in cvs), dtype=np.float64, count=n_cv)
    radius_dist = np.fromiter((cv.radius_dist.mantissa if isinstance(cv.radius_dist, u.Quantity) else float(cv.radius_dist) for cv in cvs), dtype=np.float64, count=n_cv)
    diam_arc_mean = np.fromiter((cv.diam_arc_mean.mantissa if isinstance(cv.diam_arc_mean, u.Quantity) else float(cv.diam_arc_mean) for cv in cvs), dtype=np.float64, count=n_cv)
    diam_mid = np.fromiter((2.0 * (cv.radius_mid.mantissa if isinstance(cv.radius_mid, u.Quantity) else float(cv.radius_mid)) for cv in cvs), dtype=np.float64, count=n_cv)

    point_geom = {
        "length": _fast_quantity(np.zeros(n_point, dtype=np.float64), _UM_UNIT),
        "area": _fast_quantity(np.zeros(n_point, dtype=np.float64), _UM2_UNIT),
        "radius_prox": _fast_quantity(np.zeros(n_point, dtype=np.float64), _UM_UNIT),
        "radius_dist": _fast_quantity(np.zeros(n_point, dtype=np.float64), _UM_UNIT),
        "diam_arc_mean": _fast_quantity(np.zeros(n_point, dtype=np.float64), _UM_UNIT),
        "diam_mid": _fast_quantity(np.zeros(n_point, dtype=np.float64), _UM_UNIT),
    }
    point_geom["length"]._mantissa[point_ids] = length
    point_geom["area"]._mantissa[point_ids] = area
    point_geom["radius_prox"]._mantissa[point_ids] = radius_prox
    point_geom["radius_dist"]._mantissa[point_ids] = radius_dist
    point_geom["diam_arc_mean"]._mantissa[point_ids] = diam_arc_mean
    point_geom["diam_mid"]._mantissa[point_ids] = diam_mid

    for ion in ions.values():
        pop_size = tuple(getattr(ion, "varshape", ())[:-1])
        point_shape = pop_size + (n_point,)
        for attr in ("length", "area", "diam_mid", "diam_arc_mean", "radius_prox", "radius_dist"):
            val = bridge.broadcast_to_shape(point_geom[attr], point_shape, name=f"ion.{attr}")
            setattr(ion, attr, val)


_runtime_module._instantiate_runtime_ion_instance = _fast_instantiate_runtime_ion_instance
_runtime_module.attach_runtime_ion_geometry = _fast_attach_runtime_ion_geometry
bridge.attach_runtime_ion_geometry = _fast_attach_runtime_ion_geometry


@contextmanager
def _indexed_morphology(morphology):
    original = morphology._branch_index
    original_branches = morphology.branch_by_order
    saved = {name: (name in morphology.__dict__, morphology.__dict__.get(name))
             for name in ("_branch_index", "branch_by_order")}
    stamp, indices = None, None
    branch_stamp, ordered_branches = None, None

    def index(node_id, *, order="default"):
        nonlocal stamp, indices
        if order != "default":
            return original(node_id, order=order)
        current = (len(morphology._nodes), morphology._next_id)
        if current != stamp:
            indices = morphology._branch_index_map(order="default")
            stamp = current
        return indices[node_id]

    def branches(*, order="default"):
        nonlocal branch_stamp, ordered_branches
        if order != "default":
            return original_branches(order=order)
        current = (len(morphology._nodes), morphology._next_id)
        if current != branch_stamp:
            ordered_branches = original_branches(order="default")
            branch_stamp = current
        return ordered_branches

    morphology._branch_index = index
    morphology.branch_by_order = branches
    try:
        yield
    finally:
        for name, (had_override, previous) in saved.items():
            if had_override:
                setattr(morphology, name, previous)
            else:
                delattr(morphology, name)


def build_dhs_static_source_1d(target, *, node_tree, scheduling) -> _st.DHSStaticSource:
    """Assemble the static 1D DHS source arrays without dense matrix allocation.

    Parameters
    ----------
    target : braincell.Cell
        Multi-compartment target cell.
    node_tree : braincell.NodeTree
        Morphological node tree.
    scheduling : braincell.NodeScheduling
        Node scheduling metadata.

    Returns
    -------
    _st.DHSStaticSource
        Static DHS topology and diagonal / off-diagonal factors.
    """
    n_point = len(node_tree.nodes)
    point_id_to_row = np.asarray(scheduling.point_id_to_row, dtype=np.int32)
    cv_row_by_cv = point_id_to_row[node_tree.cv_to_mid_node_id]
    n_cv = len(target.cvs)
    dynamic_rows = cv_row_by_cv[:n_cv].astype(np.int32)

    cv_r_prox = np.empty(n_cv, dtype=np.float64)
    cv_r_dist = np.empty(n_cv, dtype=np.float64)
    cv_cap = np.empty(n_cv, dtype=np.float64)
    cv_branches = np.empty(n_cv, dtype=np.int32)

    for i, cv in enumerate(target.cvs):
        cv_branches[i] = cv.branch_id
        r_p = cv.r_axial_prox
        cv_r_prox[i] = r_p.mantissa if isinstance(r_p, u.Quantity) else float(r_p)
        r_d = cv.r_axial_dist
        cv_r_dist[i] = r_d.mantissa if isinstance(r_d, u.Quantity) else float(r_d)
        area = cv.area.mantissa if isinstance(cv.area, u.Quantity) else float(cv.area)
        cm = cv.cm.mantissa if isinstance(cv.cm, u.Quantity) else float(cm)
        cv_cap[i] = (area * cm) * 1e-8

    row_capacitance_uF = np.ones(n_point, dtype=np.float64)
    row_capacitance_uF[dynamic_rows] = cv_cap

    diag_ms_inv = np.zeros(n_point, dtype=np.float64)
    lowers_ms_inv = np.zeros(n_point, dtype=np.float64)
    uppers_ms_inv = np.zeros(n_point, dtype=np.float64)

    for edge in node_tree.edges:
        parent_row = point_id_to_row[edge.parent_node_id]
        child_row = point_id_to_row[edge.child_node_id]
        roles = edge.roles
        if len(roles) == 1:
            role = roles[0]
            r = cv_r_prox[role.cv_id] if role.half == "prox" else cv_r_dist[role.cv_id]
            conductance = 1.0 / r
        else:
            b_set = {cv_branches[r.cv_id] for r in roles}
            res_list = [cv_r_prox[r.cv_id] if r.half == "prox" else cv_r_dist[r.cv_id] for r in roles]
            if len(b_set) == 1:
                conductance = 1.0 / sum(res_list)
            else:
                conductance = sum(1.0 / r for r in res_list)

        parent_coeff = (conductance / row_capacitance_uF[parent_row]) * 1e3
        child_coeff = (conductance / row_capacitance_uF[child_row]) * 1e3
        diag_ms_inv[parent_row] += parent_coeff
        diag_ms_inv[child_row] += child_coeff
        lowers_ms_inv[child_row] -= child_coeff
        uppers_ms_inv[child_row] -= parent_coeff

    parent_lookup = np.empty((n_point + 1,), dtype=np.int32)
    spurious_row = n_point
    parent_lookup[:n_point] = np.where(scheduling.parent_rows >= 0, scheduling.parent_rows, spurious_row)
    parent_lookup[spurious_row] = spurious_row
    edges, level_size = _st._build_dhs_edge_order(scheduling)
    backsub_indices = _fast_build_backsub_indices(parent_lookup, n_nodes=n_point)
    level_offsets_np = np.cumsum(np.insert(level_size, 0, 0)).astype(np.int32, copy=False)

    return _st.DHSStaticSource(
        n_point=n_point,
        dynamic_rows_np=dynamic_rows,
        row_to_point_id_np=np.asarray(scheduling.row_to_point_id, dtype=np.int32),
        row_capacitance_uF_np=row_capacitance_uF,
        diag_ms_inv_np=diag_ms_inv,
        lowers_ms_inv_np=lowers_ms_inv,
        uppers_ms_inv_np=uppers_ms_inv,
        edges_np=edges,
        level_offsets_np=level_offsets_np,
        backsub_indices_np=backsub_indices,
    )


_GLOBAL_DISCRETIZATION_CACHE = WeakValueDictionary()


class H01Cell(braincell.Cell):
    """BrainCell cell with branch-index lookup caching and sparse solver memory.

    Parameters
    ----------
    morpho : braincell.Morphology
        Source geometry, as accepted by ``braincell.Cell``.
    **kwargs
        Unchanged BrainCell cell parameters.

    Notes
    -----
    This changes index lookup work and replaces dense axial operator allocation
    during ``init_state`` with a 1D sparse DHS tree source. The ordinary discretization,
    channel mechanisms, and solver remain in use. Dense axial reduction is computed
    only on demand if ``compute_axial_derivative`` or ``_get_axial_operator`` is called.
    """

    @property
    def _discretization(self):
        sig = getattr(self._morpho, "_h01_geom_sig", None)
        if sig is not None:
            disc_key = (sig, self.cv_policy, tuple(self._paint_rules), tuple(self._place_rules))
            cached = _GLOBAL_DISCRETIZATION_CACHE.get(disc_key)
            if cached is not None:
                self._discretization_cache = cached
                self._discretization_cache_key = self._discretization_key()
                return cached
        with _indexed_morphology(self._morpho):
            res = super()._discretization
            if sig is not None:
                disc_key = (sig, self.cv_policy, tuple(self._paint_rules), tuple(self._place_rules))
                _GLOBAL_DISCRETIZATION_CACHE[disc_key] = res
            return res

    @property
    def n_cv(self) -> int:
        cache = self.__dict__.get("_discretization_cache")
        if cache is not None:
            return len(cache.cvs)
        bounds = self.cv_policy.resolve_cv_bounds(self._morpho)
        return sum(len(b) for b in bounds)

    def init_state(self, batch_size=None) -> None:
        """Lower the declaration into runtime state with sparse solver metadata.

        Parameters
        ----------
        batch_size : int, optional
            Optional batch dimension.

        Raises
        ------
        RuntimeError
            If the cell is already initialized. Call :meth:`reset` first.
        """
        self._raise_if_initialized("init_state()")

        cached_disc = self.__dict__.get("_discretization_cache")
        sig = getattr(self._morpho, "_h01_geom_sig", None)
        morpho = _fast_clone_morpho(self._morpho)
        if sig is not None:
            try:
                morpho._h01_geom_sig = sig
            except (AttributeError, TypeError):
                pass
        self._morpho = morpho
        if cached_disc is not None:
            self._discretization_cache = cached_disc
            self._discretization_cache_key = self._discretization_key()
        else:
            self._invalidate_discretization_cache()
            _ = self._discretization

        self._runtime = CellRuntimeState.from_cell(self)
        self._V_th_declaration = self._V_th

        self._in_size = self.varshape
        self._out_size = self.varshape

        root_nodes = dict(self._runtime.ions)
        for layout in self._runtime.layouts:
            node = self._runtime.runtime_nodes.get(layout.id)
            if node is None:
                continue
            if _is_root_level_runtime_node(layout.kind):
                root_nodes[f"layout_{layout.id}"] = node

        self.ion_channels = self._format_elements(IonChannel, **root_nodes)
        
        varshape = self.varshape
        v_initializer = (
            self._V_init if self._V_init is not None
            else cv_value_vector(self, attr_name="v")
        )
        if batch_size is None and not is_traced_value(v_initializer):
            n_cv = len(self.cvs)
            cm_arr = np.fromiter((cv.cm.mantissa if isinstance(cv.cm, u.Quantity) else float(cv.cm) for cv in self.cvs), dtype=np.float64, count=n_cv)
            self.C = u.Quantity(np.broadcast_to(cm_arr, varshape), u.uF / u.cm**2)
            v_th_val = self.V_th.mantissa if isinstance(self.V_th, u.Quantity) else float(self.V_th)
            self.V_th = u.Quantity(np.full(varshape, v_th_val, dtype=np.float64), u.mV)
            if isinstance(v_initializer, u.Quantity):
                v_init_val = v_initializer.mantissa
                v_unit = v_initializer.unit
            else:
                v_init_val = float(v_initializer)
                v_unit = u.mV
            v_np = np.broadcast_to(np.asarray(v_init_val, dtype=np.float64), varshape)
            self.V = DiffEqState(u.Quantity(v_np, v_unit))
            self.spike = brainstate.ShortTermState(np.zeros(varshape, dtype=np.float64))
            self._current_time_state.value = 0.0 * u.ms

            point_ids = self._runtime.node_tree.cv_to_mid_node_id
            n_point = self._runtime.n_point
            point_V_np = np.zeros(varshape[:-1] + (n_point,), dtype=np.float64)
            point_V_np[..., point_ids] = v_np
            point_V_init = u.Quantity(point_V_np, v_unit)
        else:
            self.C = cv_value_vector(self, attr_name="cm")
            self.V_th = bridge.fill_like(self.varshape, self.V_th)
            if self._V_init is not None:
                v_initializer = bridge.fill_like(self.varshape, v_initializer)
            v_value = braintools.init.param(v_initializer, self.varshape)
            v_value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
            self.V = DiffEqState(v_value)
            self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
            self._current_time_state.value = 0.0 * u.ms
            point_V = self._cv_to_point_unchecked(self.V.value)
            point_V_init = point_V

        for path, channel in self._runtime_objects_unchecked(
            IonChannel, allowed_hierarchy=(1, 1)
        ).items():
            args = self._runtime_node_phase_args(path, channel, point_V_init)
            channel.init_state(*args, batch_size=batch_size)

        scheduling = self._node_scheduling_unchecked(algorithm="dhs")
        dhs_source = build_dhs_static_source_1d(
            self,
            node_tree=self.node_tree,
            scheduling=scheduling,
        )
        self._runtime.dhs_static_source_np = dhs_source
        try:
            from .h01_dhs_scan import _prepare_levels
            dhs_cache = _st._get_dhs_static_cache(self, dhs_source)
            levels = _prepare_levels(dhs_source.edges_np, dhs_source.level_offsets_np, dhs_source.n_point)
            self._runtime.h01_dhs_pack = (dhs_source, dhs_cache, levels)
        except Exception:
            pass
        self._runtime.axial_operator_np = None
        self._runtime.axial_operator_cache = None
        self._axial_jax = None
        self._initialized = True
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None
        # Keep the live snapshot owned by this cell. Shared lookup uses weak
        # values so invalidated previews and retired cells can be collected.

    def _voltage_linearizer(self):
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            cached = getattr(runtime, "_voltage_linearizer_cache", None)
            if cached is not None:
                return cached
        def membrane_derivative(voltage, time, timestep, *args):
            with brainstate.environ.context(t=time, dt=timestep):
                return self.compute_membrane_derivative(voltage, *args)

        gradient = brainstate.transform.vector_grad(
            membrane_derivative,
            argnums=0,
            return_value=True,
            unit_aware=False,
        )
        def lin(voltage, *args):
            # A cached StatefulFunction must not close over an outer loop's
            # traced clock. Both environment values are runtime inputs here.
            return gradient(voltage, brainstate.environ.get('t', 0.*u.ms),
                            brainstate.environ.get('dt'), *args)
        if runtime is not None:
            runtime._voltage_linearizer_cache = lin
        return lin

    def _get_axial_operator(self):
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("_get_axial_operator() requires init_state() first.")
        float_dtype = jnp.asarray(0.0).dtype
        cache = runtime.axial_operator_cache
        if cache is not None and cache.float_dtype == float_dtype:
            self._axial_jax = cache.operator
            return cache.operator

        if runtime.axial_operator_np is None:
            runtime.axial_operator_np = np.asarray(
                build_cv_axial_operator(
                    self,
                    node_tree=self.node_tree,
                    scheduling=self._node_scheduling_unchecked(algorithm="dhs"),
                ),
                dtype=np.float64,
            )

        return super()._get_axial_operator()
