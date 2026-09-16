"""One BrainCell cell over a forest of H01 cells: one kernel per mechanism.

Spec: docs/specs/2026-09-16-h01-fused-population.md, section 2. The N cells are
built and initialized as today; their discretizations are concatenated
(:mod:`h01_forest_discretization`) into one ``H01ForestCell`` whose runtime holds
one node per ``(class, name)`` mechanism over every compartment of every cell.
After ``init_state`` the per-point parameter vectors (densities, reversal
potentials, gate phase factors, calcium constants) and every dynamical state
(gates, calcium, synapse conductances, voltage) are copied point for point from
the initialized source cells, so the forest starts in exactly the per-cell state
and integrates the same equations with the same donor parameters at every
compartment. The axial solve runs once over the forest
(:mod:`h01_dhs_forest`); every mechanism kernel runs once over all points.
"""

from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np
from braincell.mech._density import Density

from .h01_construction import H01Cell
from .h01_dhs_forest import build_schedule
from .h01_forest_discretization import canonical_key, fuse_discretizations
from .h01_spike_output import _SiteSpike


@dataclass(frozen=True)
class _ForestSpike:
    """Emit spikes only from the source cells' output CVs."""

    base: object
    output_cv_ids: tuple
    n_cv: int
    mask: object = None

    def __init__(self, base, output_cv_ids, n_cv):
        mask = np.zeros(int(n_cv), dtype=bool)
        mask[list(output_cv_ids)] = True
        for name, value in (('base', base), ('output_cv_ids', tuple(output_cv_ids)), ('n_cv', int(n_cv)), ('mask', mask)):
            object.__setattr__(self, name, value)

    def __call__(self, voltage):
        if voltage.shape[-1] != self.n_cv:
            raise ValueError('Forest output mask no longer matches the mesh')
        return self.base(voltage)*self.mask


def _source_layouts(cell):
    """``canonical_key -> [(layout, declaration)]`` for one initialized cell."""
    table = {}
    for layout in cell._runtime.layouts:
        declaration = cell._runtime.layout_mechanisms[layout.id]
        if layout.target == 'density' and isinstance(declaration, Density):
            table.setdefault(canonical_key(declaration), []).append((layout, declaration))
    return table


def _copy_states(source, target, point_offset, point_index):
    """Copy every state array of ``source`` into ``target`` at the offset points."""
    targets = brainstate.graph.states(target)
    for path, state in brainstate.graph.states(source).items():
        if len(path) != 1:   # a node's own states; bound children are copied through their own layouts
            continue
        if path not in targets:
            raise ValueError(f'Source state {path!r} of {type(source).__name__} has no forest counterpart')
        destination = targets[path]
        value, incoming = destination.value, state.value
        if not hasattr(u.get_mantissa(value), 'shape') or u.get_mantissa(value).shape[-1:] != (target.varshape[-1],):
            continue
        rows = point_offset+np.asarray(point_index)
        unit = u.get_unit(value)
        merged = np.array(u.get_mantissa(value), dtype=np.float64)
        merged[..., rows] = np.asarray(u.get_mantissa(incoming if unit == u.UNITLESS else incoming.in_unit(unit)))[..., np.asarray(point_index)]
        destination.value = merged if unit == u.UNITLESS else u.Quantity(merged, unit)


class H01ForestCell(H01Cell):
    """One cell whose compartments are the union of N initialized H01 cells.

    Parameters
    ----------
    cells : sequence of braincell.Cell
        Initialized source cells in population order; their discretizations,
        parameters and states are the forest's.
    solver : str, optional
        Integrator; the pinned ``h01_staggered_calcium_implicit`` detects the
        forest pack on the runtime and solves the forest once.

    Notes
    -----
    ``forest_offsets`` gives each cell's CV/point/branch offsets;
    ``soma_cv_ids`` and ``output_cv_ids`` the per-cell readout and emission CVs.
    """

    def __init__(self, cells, *, solver='h01_staggered_calcium_implicit'):
        cells = tuple(cells)
        if not cells or any(not getattr(c, '_initialized', False) for c in cells):
            raise ValueError('The forest needs at least one initialized source cell')
        forest, offsets, canonical = fuse_discretizations([c._discretization for c in cells])
        self._forest = forest
        self.forest_offsets = offsets
        self._canonical = canonical
        self._source_cells = cells
        v_init = np.concatenate([np.asarray(c.V.value.to_decimal(u.mV)).reshape(-1) for c in cells])
        outputs = []
        for index, cell in enumerate(cells):
            if not isinstance(cell.spk_fun, _SiteSpike):
                raise ValueError('Every source cell must emit from one restricted output site')
            outputs.append(int(cell.spk_fun.cv_id)+int(offsets.cv[index]))
        self.output_cv_ids = tuple(outputs)
        self.contacts = ()
        super().__init__(cells[0].morpho, cv_policy=cells[0].cv_policy, V_init=v_init*u.mV,
                         V_th=cells[0]._V_th_declaration, solver=solver, pop_size=(1,), name='forest')
        self.spk_fun = _ForestSpike(cells[0].spk_fun.base, self.output_cv_ids, len(forest.cvs))
        self._discretization_cache = forest
        self.soma_cv_ids = tuple(int(o)+_soma_cv(c) for o, c in zip(offsets.cv[:-1], cells))

    @property
    def _discretization(self):
        return self._forest

    @property
    def cvs(self):
        return self._forest.cvs

    @property
    def n_cv(self):
        return len(self._forest.cvs)

    def init_state(self, batch_size=None):
        """Lower the forest, then copy per-point parameters and states from the sources.

        Parameters
        ----------
        batch_size : int, optional
            Unsupported; the forest is one population.
        """
        if batch_size is not None:
            raise ValueError('A forest cell has no batch axis')
        super().init_state()
        runtime = self._runtime
        source = runtime.dhs_static_source_np
        runtime.h01_dhs_forest = build_schedule(source.edges_np, source.n_point)
        tables = [_source_layouts(cell) for cell in self._source_cells]
        for layout in runtime.layouts:
            declaration = runtime.layout_mechanisms[layout.id]
            if layout.target != 'density' or not isinstance(declaration, Density):
                continue
            key = canonical_key(declaration)
            self._write_parameters(layout, declaration, key, tables)
            self._copy_layout_states(layout, key, tables)
        for name, ion in runtime.ions.items():
            for index, cell in enumerate(self._source_cells):
                if name in cell._runtime.ions:
                    _copy_states(cell._runtime.ions[name], ion, int(self.forest_offsets.point[index]),
                                 np.arange(cell._runtime.n_point))

    def _write_parameters(self, layout, declaration, key, tables):
        runtime = self._runtime
        node = runtime.runtime_nodes.get(layout.id) or runtime.ions.get(declaration.instance_name)
        for var in declaration.params.keys():
            buffer = runtime.state_buffers[(layout.id, var)]
            unit = buffer.unit if isinstance(buffer, u.Quantity) else None
            vector = np.full(np.shape(u.get_mantissa(buffer)), _default_value(type(node), var, unit), dtype=np.float64)
            for index, table in enumerate(tables):
                offset = int(self.forest_offsets.point[index])
                for source_layout, source_declaration in table.get(key, ()):
                    if var not in source_declaration.params:
                        continue   # the source used the class default, already in place
                    source = self._source_cells[index]._runtime.state_buffers[(source_layout.id, var)]
                    values = np.asarray(u.get_mantissa(source if unit is None else source.in_unit(unit)))
                    vector[..., offset+source_layout.point_index] = values[..., source_layout.point_index]
            value = vector if unit is None else u.Quantity(vector, unit)
            if declaration.category == 'ion':
                # BrainCell's ion re-sync ignores the population axis; write the attribute directly.
                runtime.state_buffers[(layout.id, var)] = value
                ion = runtime.ions[declaration.instance_name]
                setattr(ion, var, value)
                if callable(getattr(ion, '_update_reversal', None)):
                    ion._update_reversal()
            else:
                runtime.set_state(layout.id, var, value)

    def _copy_layout_states(self, layout, key, tables):
        node = self._runtime.runtime_nodes.get(layout.id)
        if node is None:
            return
        for index, table in enumerate(tables):
            offset = int(self.forest_offsets.point[index])
            for source_layout, _ in table.get(key, ()):
                source_node = self._source_cells[index]._runtime.runtime_nodes.get(source_layout.id)
                if source_node is not None:
                    _copy_states(source_node, node, offset, source_layout.point_index)


def _default_value(node_cls, var, unit):
    """Constructor default of ``var`` for ``node_cls`` in ``unit`` (0 when it has none)."""
    import inspect
    parameter = inspect.signature(node_cls.__init__).parameters.get(var)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return 0.
    default = parameter.default
    if isinstance(default, u.Quantity):
        return float(default.to_decimal(unit))
    return float(default)


def _soma_cv(cell):
    """CV id of the source cell's ``voltage`` probe (its measured soma)."""
    from braincell._multi_compartment.probes import _representative_cv_id
    for layout in cell._runtime.layouts:
        declaration = cell._runtime.get_layout_mechanism(layout.id)
        if getattr(declaration, 'name', None) == 'voltage':
            return _representative_cv_id(cell._runtime, point_id=int(layout.point_index[0]))
    raise ValueError('Every source cell must carry a soma voltage probe')
