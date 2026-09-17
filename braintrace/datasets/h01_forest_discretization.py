"""Concatenate per-cell BrainCell discretizations into one forest discretization.

Spec: docs/specs/2026-09-16-h01-fused-population.md, section 2. ``Discretization``,
``CVTree`` and ``NodeTree`` are plain frozen dataclasses and the runtime reads
only ``cell.cvs``/``cell.node_tree``, so a forest is one ``Discretization`` whose
CV, node, edge and branch ids are the per-cell ids plus cumulative offsets.
Every CV keeps its own geometry, cable properties, resting potential and point
mechanisms. Density declarations are replaced by one canonical declaration per
``(category, class, name)`` so that the runtime builds one node per mechanism
over all compartments; the per-point parameter vectors are written after
``init_state`` (see :mod:`h01_forest_cell`).
"""

from dataclasses import dataclass, replace

import numpy as np
from braincell._discretization.base import (
    CVTree, Discretization, Node, NodeEdge, NodeEdgeRole, NodeRole, NodeTree, CVEdge,
)
from braincell.mech import Channel, Ion


@dataclass(frozen=True)
class ForestOffsets:
    """Row offsets of each cell inside the forest.

    Attributes
    ----------
    cv : numpy.ndarray
        ``(cells+1,)`` cumulative CV counts.
    point : numpy.ndarray
        ``(cells+1,)`` cumulative point (node) counts.
    branch : numpy.ndarray
        ``(cells+1,)`` cumulative branch counts.
    """

    cv: np.ndarray
    point: np.ndarray
    branch: np.ndarray

    @property
    def cells(self):
        return len(self.cv)-1


def canonical_key(declaration):
    """Identity of the canonical layout a density declaration folds into.

    Parameters
    ----------
    declaration : Density
        A ``Channel`` or ``Ion`` declaration.

    Returns
    -------
    tuple
        ``(category, class_name, instance_name)``.
    """
    return (declaration.category, declaration.class_name, declaration.instance_name)


def canonical_declarations(discretizations):
    """One canonical declaration per ``(category, class, name)`` over all cells.

    Parameters
    ----------
    discretizations : sequence of Discretization
        Per-cell discretizations.

    Returns
    -------
    dict
        ``canonical_key -> declaration`` where the declaration carries the union
        of the parameter names seen for that key (values from the first
        occurrence: placeholders, overwritten per point after ``init_state``),
        full coverage and the first occurrence's ion selectors.
    """
    seen = {}
    for discretization in discretizations:
        for cv in discretization.cvs:
            for declaration in cv.density_mech:
                first, params = seen.setdefault(canonical_key(declaration), (declaration, {}))
                for name, value in declaration.params.items():
                    params.setdefault(name, value)
    canonical = {}
    for key, (first, params) in seen.items():
        if first.category == 'channel':
            canonical[key] = Channel(first.class_name, name=first.name, ion_name=first.ion_name,
                                     ion_names=first.ion_names, **params)
        else:
            canonical[key] = Ion(first.class_name, name=first.name, **params)
    return canonical


def _canonical_tuple(declarations, canonical):
    out, seen = [], set()
    for declaration in declarations:
        key = canonical_key(declaration)
        if key not in seen:
            seen.add(key)
            out.append(canonical[key])
    return tuple(out)


def fuse_discretizations(discretizations):
    """Concatenate per-cell discretizations into one forest.

    Parameters
    ----------
    discretizations : sequence of Discretization
        Per-cell discretizations in population order.

    Returns
    -------
    tuple
        ``(forest, offsets, canonical)``: the forest ``Discretization``, the
        :class:`ForestOffsets`, and the canonical declaration table.
    """
    if not discretizations:
        raise ValueError('A forest needs at least one cell')
    canonical = canonical_declarations(discretizations)
    counts = np.asarray([[len(d.cvs), len(d.nodes), len(d.cv_tree.branch_to_cv_ids)] for d in discretizations])
    offsets = ForestOffsets(*(np.concatenate([[0], np.cumsum(counts[:, i])]).astype(np.int64) for i in range(3)))
    cvs, cv_edges, branches, nodes, node_edges, mid, endpoints = [], [], [], [], [], [], []
    edge_offset = 0
    for index, discretization in enumerate(discretizations):
        cv_off, pt_off, br_off = (int(o[index]) for o in (offsets.cv, offsets.point, offsets.branch))
        for cv in discretization.cvs:
            cvs.append(replace(cv, id=cv.id+cv_off, branch_id=cv.branch_id+br_off,
                parent_cv=None if cv.parent_cv is None else cv.parent_cv+cv_off,
                children_cv=tuple(c+cv_off for c in cv.children_cv),
                density_mech=_canonical_tuple(cv.density_mech, canonical)))
        cv_edges.extend(CVEdge(e.parent_cv_id+cv_off, e.child_cv_id+cv_off) for e in discretization.cv_tree.edges)
        branches.extend(tuple(c+cv_off for c in ids) for ids in discretization.cv_tree.branch_to_cv_ids)
        tree = discretization.node_tree
        for node in tree.nodes:
            nodes.append(replace(node, id=node.id+pt_off,
                roles=tuple(NodeRole(r.cv_id+cv_off, r.position) for r in node.roles),
                density_mech=_canonical_tuple(node.density_mech, canonical)))
        for edge in tree.edges:
            node_edges.append(NodeEdge(edge.id+edge_offset, edge.parent_node_id+pt_off, edge.child_node_id+pt_off,
                tuple(NodeEdgeRole(r.cv_id+cv_off, r.half, r.r_axial) for r in edge.roles)))
        edge_offset += len(tree.edges)
        mid.append(np.asarray(tree.cv_to_mid_node_id)+pt_off)
        endpoints.append(np.asarray(tree.branch_endpoint_node_id)+pt_off)
    cv_tree = CVTree(cvs=tuple(cvs), edges=tuple(cv_edges), root_cv_id=0, branch_to_cv_ids=tuple(branches))
    node_tree = NodeTree(nodes=tuple(nodes), edges=tuple(node_edges), root_node_id=int(discretizations[0].node_tree.root_node_id),
        cv_to_mid_node_id=np.concatenate(mid), branch_endpoint_node_id=np.concatenate(endpoints))
    return Discretization(cv_tree=cv_tree, node_tree=node_tree), offsets, canonical
