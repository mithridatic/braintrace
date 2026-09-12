"""Explicit spine additions with geometry-preserving cable subdivision."""

from dataclasses import dataclass

import braincell
import brainunit as u
import numpy as np


@dataclass(frozen=True)
class Spine:
    """One immutable cylindrical neck/head addition.

    Parameters
    ----------
    identity : str
        Unique addition identity, distinct from measured source identifiers.
    parent_branch : int
        Original morphology branch index.
    parent_x : float
        Normalized original cable attachment position.
    neck_length_um, neck_radius_um, head_length_um, head_radius_um : float
        Positive cylinder dimensions. Heads are explicit cylinders, not spheres.
    direction : tuple of float
        Unit direction for spatial extension from the parent cable centreline.
    provenance : str
        Source record or explicit synthetic basis. Never inferred from geometry.
    """

    identity: str
    parent_branch: int
    parent_x: float
    neck_length_um: float
    neck_radius_um: float
    head_length_um: float
    head_radius_um: float
    direction: tuple
    provenance: str

    def __post_init__(self):
        dimensions = (self.neck_length_um, self.neck_radius_um, self.head_length_um, self.head_radius_um)
        if not isinstance(self.identity, str) or not self.identity or not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("Spine identity and provenance are required")
        if type(self.parent_branch) is not int or self.parent_branch < 0 or not np.isfinite(self.parent_x) or not 0 <= self.parent_x <= 1:
            raise ValueError("Invalid spine parent attachment")
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("Spine dimensions must be positive and finite")
        direction = np.asarray(self.direction)
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1., atol=1e-10):
            raise ValueError("Spine direction must be a finite unit vector")


def _slice(branch, lo, hi):
    lengths = np.asarray(branch.lengths.to_decimal(u.um))
    arcs = np.r_[0., np.cumsum(lengths)]
    if np.any(lengths <= 0):
        raise ValueError("Subdivision requires positive original segment lengths")
    start, stop = lo*arcs[-1], hi*arcs[-1]
    a, b = np.maximum(arcs[:-1], start), np.minimum(arcs[1:], stop)
    keep = b > a
    ta, tb = (a[keep]-arcs[:-1][keep])/lengths[keep], (b[keep]-arcs[:-1][keep])/lengths[keep]
    proximal = np.asarray(branch.radii_proximal.to_decimal(u.um))[keep]
    distal = np.asarray(branch.radii_distal.to_decimal(u.um))[keep]
    kwargs = {}
    if branch.points_proximal is not None and branch.points_distal is not None:
        p = np.asarray(branch.points_proximal.to_decimal(u.um))[keep]
        q = np.asarray(branch.points_distal.to_decimal(u.um))[keep]
        kwargs = dict(points_proximal=(p+ta[:, None]*(q-p))*u.um,
                      points_distal=(p+tb[:, None]*(q-p))*u.um)
    return braincell.Branch(lengths=(b[keep]-a[keep])*u.um,
        radii_proximal=(proximal+ta*(distal-proximal))*u.um,
        radii_distal=(proximal+tb*(distal-proximal))*u.um, type=branch.type, **kwargs)


def add_spines(morphology, spines):
    """Return a new morphology and explicit source-coordinate remapping.

    Parameters
    ----------
    morphology : braincell.Morphology
        Unmodified source cable tree.
    spines : sequence of Spine
        Explicit additions; previously modeled spines must not be supplied again.

    Returns
    -------
    tuple
        New morphology, source mapping ``{index: [(lo, hi, new_name), ...]}``,
        and added head names keyed by spine identity. Mapping retains physical
        cable intervals even when original branch traversal is reversed.

    Notes
    -----
    Splits interpolate existing tapered segments exactly; no original membrane
    is removed or replaced. Consumers must remap regions, probes and contacts
    through this mapping instead of reusing original branch indices.
    """
    additions = tuple(spines)
    if len({s.identity for s in additions}) != len(additions):
        raise ValueError("Duplicate spine identity")
    branches = morphology.branches
    if any(s.parent_branch >= len(branches) for s in additions):
        raise ValueError("Spine parent is outside source morphology")
    cuts = {i: {0., 1.} for i in range(len(branches))}
    for s in additions:
        cuts[s.parent_branch].add(float(s.parent_x))
    parents = {}
    for edge in morphology.edges:
        parents[edge.child.index] = (edge.parent.index, edge.parent_x, edge.child_x)
        cuts[edge.parent.index].add(float(edge.parent_x))
    mapping = {i: [(a, b, f'source_{i}_{j}') for j, (a, b) in enumerate(zip(sorted(cuts[i])[:-1], sorted(cuts[i])[1:]))]
               for i in range(len(branches))}

    def locate(index, x):
        for lo, hi, name in mapping[index]:
            if lo <= x <= hi:
                return name, 0. if x == lo else 1.
        raise ValueError("Attachment cannot be mapped")

    pending, built, result = set(mapping), set(), None
    # Static topology traversal only.
    while pending:
        ready = sorted(i for i in pending if i not in parents or parents[i][0] in built)
        if not ready:
            raise ValueError("Source morphology contains a cycle")
        for index in ready:
            reverse = index in parents and parents[index][2] == 1.
            pieces = list(reversed(mapping[index])) if reverse else mapping[index]
            previous = None
            for lo, hi, name in pieces:
                branch = _slice(branches[index].branch, lo, hi)
                if index not in parents and previous is None:
                    result = braincell.Morphology(root_name=name, root_branch=branch)
                else:
                    if previous is not None:
                        parent, x = previous, 0. if reverse else 1.
                    else:
                        source_parent, px, _ = parents[index]
                        parent, x = locate(source_parent, px)
                    result.attach(parent=parent, child_name=name, child_branch=branch,
                                  parent_x=x, child_x=1. if reverse else 0.)
                previous = name
            built.add(index)
            pending.remove(index)
    heads = {}
    views = {view.name: view for view in result.branches}
    for spine in additions:
        parent, x = locate(spine.parent_branch, spine.parent_x)
        view = views[parent]
        point_array = view.branch.points_proximal if x == 0 else view.branch.points_distal
        if point_array is None:
            raise ValueError("Spatial spine placement requires source xyz coordinates")
        point = np.asarray(point_array.to_decimal(u.um))[0 if x == 0 else -1]
        for part, length, radius in (('neck', spine.neck_length_um, spine.neck_radius_um),
                                     ('head', spine.head_length_um, spine.head_radius_um)):
            end = point + np.asarray(spine.direction)*length
            name = 'added_'+spine.identity+'_'+part
            branch = braincell.Branch(lengths=np.array([length])*u.um,
                radii_proximal=np.array([radius])*u.um, radii_distal=np.array([radius])*u.um,
                points_proximal=point[None, :]*u.um, points_distal=end[None, :]*u.um, type='dendrite')
            result.attach(parent=parent, child_name=name, child_branch=branch, parent_x=x)
            parent, x, point = name, 1., end
        heads[spine.identity] = name
    return result, mapping, heads
