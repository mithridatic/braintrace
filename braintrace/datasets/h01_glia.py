"""Source-pinned, unjoined H01 glial skeleton fragments in physical coordinates."""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import braincell
import brainunit as u
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .h01_anatomy import _geometry_signature


@dataclass(frozen=True, init=False)
class H01GlialSelection:
    """Freeze an explicit component selection against an acquired skeleton.

    Parameters
    ----------
    document : dict
        Exact h01-glial-fragment-v1 declaration: source_sha256, identity,
        source, attribution, coordinate_frame, anchor_vertex, max_vertices
        and selection_basis. An anchor selects its entire connected component.
    """

    _json: str

    def __init__(self, document):
        required = {'schema', 'source_sha256', 'identity', 'source', 'attribution',
                    'coordinate_frame', 'anchor_vertex', 'max_vertices', 'selection_basis'}
        if not isinstance(document, dict) or set(document) != required:
            raise ValueError('Glial selection requires exact manifest fields')
        if document['schema'] != 'h01-glial-fragment-v1' or document['coordinate_frame'] != 'h01-c3-nm':
            raise ValueError('Unsupported glial schema or coordinate frame')
        digest = document['source_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Glial source requires a SHA256 digest')
        if any(not isinstance(document[key], str) or not document[key].strip()
               for key in ('identity', 'source', 'attribution', 'selection_basis')):
            raise ValueError('Source identity, attribution and selection basis are required')
        if (type(document['anchor_vertex']) is not int or document['anchor_vertex'] < 0 or
                type(document['max_vertices']) is not int or document['max_vertices'] < 2):
            raise ValueError('Explicit anchor and positive component size limit required')
        object.__setattr__(self, '_json', json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False))

    def to_dict(self):
        """Return a detached canonical manifest document.

        Returns
        -------
        dict
            Editable copy; the frozen selection remains unchanged.
        """
        return json.loads(self._json)

    @property
    def sha256(self):
        """Return the immutable canonical selection identity."""
        return hashlib.sha256(self._json.encode()).hexdigest()


@dataclass(frozen=True)
class H01GlialFragment:
    """Preserved source morphology and detached reconstruction evidence.

    Attributes
    ----------
    morphology : braincell.Morphology
        Complete selected component, without added soma or fragment bridges.
    evidence_json : str
        Canonical source identities, source paths and excluded inventory.
    """

    morphology: object
    evidence_json: str

    @property
    def evidence(self):
        """Return a detached reconstruction record."""
        return json.loads(self.evidence_json)


def load_glial_fragment(path, selection):
    """Validate a source forest and reconstruct one complete tapered component.

    Parameters
    ----------
    path : path-like
        Acquired npz with vertices_nm, radius_nm and edges arrays.
    selection : H01GlialSelection
        Immutable artifact and component selection.

    Returns
    -------
    H01GlialFragment
        Measured source geometry in um plus explicit exclusion evidence.

    Notes
    -----
    Vertices are original array indices. Source radii and all edge segments are
    retained. Sealed fragment ends are a later cable modeling assumption.
    A measured skeleton does not establish a complete biological astrocyte.
    """
    spec = selection.to_dict()
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['source_sha256']:
        raise ValueError('Glial source artifact digest differs')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'vertices_nm', 'radius_nm', 'edges'}:
            raise ValueError('Source skeleton requires exact geometry arrays')
        points = np.asarray(arrays['vertices_nm'], dtype=float)/1000.
        radii = np.asarray(arrays['radius_nm'], dtype=float)/1000.
        edges = np.asarray(arrays['edges'])
    if points.ndim != 2:
        raise ValueError('Source coordinates must be a vertex-by-xyz array')
    n = len(points)
    if (points.shape != (n, 3) or n < 2 or radii.shape != (n,) or
            not np.isfinite(points).all() or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError('Finite coordinates and positive source radii required')
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer) or
            np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('Invalid source edge indices')
    canonical = np.sort(edges, axis=1)
    if len(np.unique(canonical, axis=0)) != len(edges):
        raise ValueError('Duplicate undirected source edges')
    if np.any(np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1) <= 0):
        raise ValueError('Zero-length source edges cannot form a cable')
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    count, labels = connected_components(graph, directed=False)
    if len(edges) != n-count:
        raise ValueError('Source skeleton must be a forest without cycles')
    anchor = spec['anchor_vertex']
    if anchor >= n:
        raise ValueError('Anchor outside source skeleton')
    chosen = np.flatnonzero(labels == labels[anchor])
    if not 2 <= len(chosen) <= spec['max_vertices']:
        raise ValueError('Selected component is isolated or exceeds the declared size limit')
    adjacency = {int(i): [] for i in chosen}
    edge_ids = {}
    for index, (left, right) in enumerate(edges):
        if labels[left] == labels[anchor]:
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
            edge_ids[tuple(sorted((int(left), int(right))))] = index
    pending = deque((anchor, neighbor, None, 0.) for neighbor in sorted(adjacency[anchor]))
    morphology, paths, root_name = None, {}, None
    # Static tree construction only; time evolution uses BrainState transforms.
    while pending:
        start, following, parent, parent_x = pending.popleft()
        nodes = [start, following]
        while len(adjacency[nodes[-1]]) == 2:
            nodes.append(next(i for i in adjacency[nodes[-1]] if i != nodes[-2]))
        left, right = np.asarray(nodes[:-1]), np.asarray(nodes[1:])
        name = 'source_'+str(len(paths))
        branch = braincell.Branch(lengths=np.linalg.norm(points[right]-points[left], axis=1)*u.um,
            radii_proximal=radii[left]*u.um, radii_distal=radii[right]*u.um,
            points_proximal=points[left]*u.um, points_distal=points[right]*u.um, type='dendrite')
        if morphology is None:
            morphology = braincell.Morphology(root_name=name, root_branch=branch)
            root_name = name
        else:
            morphology.attach(parent=parent or root_name, child_name=name, child_branch=branch, parent_x=parent_x)
        paths[name] = dict(vertices=nodes, edges=[edge_ids[tuple(sorted(pair))] for pair in zip(nodes[:-1], nodes[1:])])
        for neighbor in sorted(adjacency[nodes[-1]]):
            if neighbor != nodes[-2]:
                pending.append((nodes[-1], neighbor, name, 1.))
    sizes = np.bincount(labels)
    roots = np.full(count, n)
    np.minimum.at(roots, labels, np.arange(n))
    evidence = dict(selection=spec, selection_sha256=selection.sha256, geometry_sha256=_geometry_signature(morphology),
        coordinate_scale_um=.001, selected_vertices=chosen.tolist(), branch_paths=paths,
        selected_edges=sorted(edge_ids.values()), component_count=count,
        excluded_components=[dict(anchor_vertex=int(roots[i]), vertices=int(sizes[i]))
                             for i in range(count) if i != labels[anchor]],
        qualification='Measured source fragment geometry; no completeness, synapse or extracellular-neighborhood claim')
    return H01GlialFragment(morphology, json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False))
