"""Measure anatomical geometry without turning unmatched areas into physiology."""

import io

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _scale(value):
    scale = np.asarray(value, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Position scale must contain three finite positive values.')
    return scale


def read_swc(text):
    """Read SWC columns without modifying labels, radii or coordinates.

    Parameters
    ----------
    text : str
        Seven-column SWC text, with optional comment lines.

    Returns
    -------
    numpy.ndarray
        Finite floating-point rows; topology is checked by ``cable_geometry``.
    """
    lines = [line for line in text.splitlines() if line.strip() and
             not line.lstrip().startswith('#')]
    if not lines:
        raise ValueError('SWC is empty.')
    rows = np.loadtxt(io.StringIO('\n'.join(lines)), ndmin=2)
    if rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError('SWC must have seven finite columns.')
    return rows


def _tree(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7 or not len(rows):
        raise ValueError('SWC must have nonempty seven-column rows.')
    if not np.isfinite(rows).all() or (rows[:, 5] <= 0).any():
        raise ValueError('SWC must be finite with positive radii.')
    ints = rows[:, [0, 1, 6]]
    if not np.equal(ints, np.floor(ints)).all() or np.abs(ints).max() > 2**53:
        raise ValueError('SWC identifiers and labels must be exact integers.')
    ids, labels, parents = ints.astype(np.int64).T
    roots = np.flatnonzero(parents == -1)
    if (ids < 0).any() or len(np.unique(ids)) != len(ids) or len(roots) != 1:
        raise ValueError('SWC needs unique nonnegative IDs and exactly one root.')
    lookup = {node: i for i, node in enumerate(ids)}
    child = np.flatnonzero(parents != -1)
    try:
        parent = np.array([lookup[node] for node in parents[child]], dtype=np.int64)
    except KeyError as exc:
        raise ValueError('SWC parent is absent.') from exc
    graph = coo_matrix((np.ones(len(child)), (child, parent)), shape=(len(ids), len(ids)))
    count = connected_components(graph, directed=False, return_labels=False)
    if count != 1:
        raise ValueError('SWC contains a detached cycle or component.')
    return rows, labels, parent, child


def cable_geometry(rows, *, position_um=(1., 1., 1.), radius_um=1.,
                   soma_sphere_radius_um=None):
    """Measure lateral frustum area, retaining soma and annotation ambiguities.

    Parameters
    ----------
    rows : array_like
        SWC rows for one rooted component; labels retain source meaning.
    position_um : tuple of float, optional
        Per-axis micrometres per position unit.
    radius_um : float, optional
        Micrometres per radius unit, not per diameter unit.
    soma_sphere_radius_um : float or None, optional
        Explicit model sphere radius. Its area is reported separately, never
        automatically added to cable area or inferred from soma skeleton nodes.

    Returns
    -------
    dict
        Geometry totals and per-endpoint-label groups. Zero-length edges are
        counted but excluded from lateral area. No end caps are invented.
    """
    rows, labels, parent, child = _tree(rows)
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError('Radius scale must be finite and positive.')
    if soma_sphere_radius_um is not None and (
            not np.isfinite(soma_sphere_radius_um) or soma_sphere_radius_um <= 0):
        raise ValueError('Explicit soma sphere radius must be positive and finite.')
    positions = rows[:, 2:5] * _scale(position_um)
    radii = rows[:, 5] * radius_um
    length = np.linalg.norm(positions[child] - positions[parent], axis=1)
    slant = np.hypot(length, radii[child] - radii[parent])
    area = np.where(length > 0, np.pi * (radii[parent] + radii[child]) * slant, 0.)
    pairs = np.column_stack((labels[parent], labels[child]))
    groups = {}
    for pair in np.unique(pairs, axis=0):
        selected = np.all(pairs == pair, axis=1)
        groups[f'{pair[0]}->{pair[1]}'] = {
            'edges': int(selected.sum()), 'length_um': float(length[selected].sum()),
            'lateral_area_um2': float(area[selected].sum())}
    return {
        'nodes': len(rows), 'edges': len(child),
        'length_um': float(length.sum()), 'lateral_area_um2': float(area.sum()),
        'zero_length_edges': int((length == 0).sum()),
        'cross_label_edges': int((labels[child] != labels[parent]).sum()),
        'radius_min_um': float(radii.min()), 'radius_max_um': float(radii.max()),
        'bounds_um': [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        'by_endpoint_labels': groups,
        'soma_sphere_area_um2': (None if soma_sphere_radius_um is None else
                                 float(4 * np.pi * soma_sphere_radius_um**2)),
        'area_convention': 'lateral frusta; no end caps; zero-length edges excluded',
    }


def mesh_geometry(vertices, faces, *, position_um=(1., 1., 1.)):
    """Measure a decoded triangle mesh with exact seam and topology diagnostics.

    Parameters
    ----------
    vertices : array_like
        Finite (N, 3) positions after the format's LOD and affine decoding.
    faces : array_like
        Integer (M, 3) indices into vertices.
    position_um : tuple of float, optional
        Per-axis micrometres per decoded position unit. Use (0.001,)*3 for
        CloudVolume meshes already transformed into physical nanometres.

    Returns
    -------
    dict
        Unique, nondegenerate triangle area and topology diagnostics. Exact
        coordinate deduplication is reported, not treated as source proof of
        connectedness. No anatomical correspondence is inferred from this area.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError('Vertices must have nonempty finite shape (N, 3).')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.isfinite(faces).all() or not np.equal(faces, np.floor(faces)).all()
            or (faces < 0).any() or (faces >= len(vertices)).any()):
        raise ValueError('Faces must be nonempty triangles with valid integer indices.')
    positions = vertices * _scale(position_um)
    unique, inverse = np.unique(positions, axis=0, return_inverse=True)
    triangles = inverse[faces.astype(np.int64)]
    corners = unique[triangles]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                    corners[:, 2] - corners[:, 0]), axis=1) / 2
    valid = areas > 0
    _, keep = np.unique(np.sort(triangles[valid], axis=1), axis=0, return_index=True)
    triangles, areas = triangles[valid][keep], areas[valid][keep]
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    used = np.unique(triangles)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(unique), len(unique)))
    _, labels = connected_components(graph, directed=False)
    component_ids = np.unique(labels[used])
    area_by_component = np.bincount(labels[triangles[:, 0]], weights=areas,
                                    minlength=len(unique))
    components = area_by_component[component_ids].tolist()
    return {
        'vertices': len(vertices), 'faces': len(faces), 'unique_vertices': len(unique),
        'duplicate_vertices': len(vertices) - len(unique),
        'degenerate_faces': int((~valid).sum()),
        'duplicate_faces': int(valid.sum()) - len(keep),
        'unused_unique_vertices': len(unique) - len(used),
        'area_um2': float(areas.sum()), 'boundary_edges': int((counts == 1).sum()),
        'nonmanifold_edges': int((counts > 2).sum()), 'components': len(components),
        'component_areas_um2': sorted(components, reverse=True),
        'bounds_um': [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        'deduplication': 'exact coordinates only; no tolerance weld',
    }
