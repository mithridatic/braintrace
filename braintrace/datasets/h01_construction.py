"""Bound branch-index work and eliminate dense axial solver overhead during H01 construction."""

from contextlib import contextmanager

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base import IonChannel
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
    dynamic_rows = np.asarray([int(cv_row_by_cv[cv_id]) for cv_id in range(len(target.cvs))], dtype=np.int32)
    row_capacitance = _st._row_capacitance_scale(target, dynamic_rows=dynamic_rows, n_point=n_point)
    row_capacitance_uF = np.asarray(
        [_st._scalar_decimal(value, u.uF) for value in row_capacitance],
        dtype=np.float64,
    )
    diag_ms_inv = np.zeros(n_point, dtype=np.float64)
    lowers_ms_inv = np.zeros(n_point, dtype=np.float64)
    uppers_ms_inv = np.zeros(n_point, dtype=np.float64)

    for edge in node_tree.edges:
        parent_row = int(point_id_to_row[edge.parent_node_id])
        child_row = int(point_id_to_row[edge.child_node_id])
        conductance = _st._edge_conductance(edge=edge, cvs=target.cvs)
        parent_coeff = _st._scalar_decimal(conductance / row_capacitance[parent_row], u.ms ** -1)
        child_coeff = _st._scalar_decimal(conductance / row_capacitance[child_row], u.ms ** -1)
        diag_ms_inv[parent_row] += parent_coeff
        diag_ms_inv[child_row] += child_coeff
        lowers_ms_inv[child_row] -= child_coeff
        uppers_ms_inv[child_row] -= parent_coeff

    parent_lookup = np.empty((n_point + 1,), dtype=np.int32)
    spurious_row = n_point
    parent_lookup[:n_point] = np.where(scheduling.parent_rows >= 0, scheduling.parent_rows, spurious_row)
    parent_lookup[spurious_row] = spurious_row
    edges, level_size = _st._build_dhs_edge_order(scheduling)
    backsub_indices = _st._build_backsub_indices(parent_lookup, n_nodes=n_point)
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
        with _indexed_morphology(self._morpho):
            return super()._discretization

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

        morpho = clone_morpho(self._morpho)
        self._morpho = morpho
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
        self.C = cv_value_vector(self, attr_name="cm")
        self.V_th = bridge.fill_like(self.varshape, self.V_th)

        v_initializer = (
            self._V_init if self._V_init is not None
            else cv_value_vector(self, attr_name="v")
        )
        if self._V_init is not None:
            v_initializer = bridge.fill_like(self.varshape, v_initializer)
        v_value = braintools.init.param(v_initializer, self.varshape)
        v_value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        self.V = DiffEqState(v_value)
        self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
        self._current_time_state.value = 0.0 * u.ms

        point_V = self._cv_to_point_unchecked(self.V.value)
        for path, channel in self._runtime_objects_unchecked(
            IonChannel, allowed_hierarchy=(1, 1)
        ).items():
            args = self._runtime_node_phase_args(path, channel, point_V)
            channel.init_state(*args, batch_size=batch_size)

        scheduling = self._node_scheduling_unchecked(algorithm="dhs")
        self._runtime.dhs_static_source_np = build_dhs_static_source_1d(
            self,
            node_tree=self.node_tree,
            scheduling=scheduling,
        )
        self._runtime.axial_operator_np = None
        self._runtime.axial_operator_cache = None
        self._axial_jax = None
        self._initialized = True
        self._runtime_cvs_cache = self._build_runtime_cv_views()
        self._runtime_nodes_cache = self._build_runtime_node_views()

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
