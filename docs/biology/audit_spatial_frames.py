"""Compare c3 and proofread coordinates without joining or attaching anatomy."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets._h01_swc import _parse_swc_bytes, POSITION_UM


def _skeleton(path):
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['coordinate_units'] != 'nm' or hashlib.sha256(path.read_bytes()).hexdigest() != metadata['artifact_sha256']:
        raise ValueError('Skeleton units or artifact hash differ')
    with np.load(path, allow_pickle=False) as data:
        points, edges = np.asarray(data['vertices_nm'], dtype=float)/1000., data['edges']
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite skeleton coordinates')
    return points, edges, metadata


def main():
    """Save same-identity coordinate diagnostics and candidate glia proximity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--reference-skeleton', type=Path, required=True)
    parser.add_argument('--glia-skeleton', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = H01Archive(args.archive)
    reference, _, reference_meta = _skeleton(args.reference_skeleton)
    glia, edges, glia_meta = _skeleton(args.glia_skeleton)
    _, components = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(glia), len(glia))).tocsr(), directed=False)
    tree, proximity, reference_points = cKDTree(glia), [], []
    inventory = json.loads(args.inventory.read_text())
    with zipfile.ZipFile(args.archive) as zipped:
        # Static source inspection only; no model evolution occurs in this loop.
        for row in inventory['cells']:
            identity = row['cell_id']
            selected = row.get('selected_component', {}).get('component', row['largest_component'])
            blob = zipped.read(archive._members[(identity, selected)])
            digest = hashlib.sha256(blob).hexdigest()
            if 'selected_component' in row and digest != row['selected_component']['source_sha256']:
                raise ValueError('Selected source component digest differs')
            points = _parse_swc_bytes(blob)[:, 2:5]*POSITION_UM
            distances, indices = tree.query(points)
            nearest = int(np.argmin(distances))
            proximity.append(dict(identity=identity, component=selected, source_sha256=digest,
                source_nodes=len(points), minimum_centerline_vertex_distance_um=float(distances[nearest]),
                nearest_source_vertex=nearest, nearest_glia_vertex=int(indices[nearest]),
                nearest_glia_component=int(components[indices[nearest]])))
        for component in archive.components(reference_meta['identity']):
            blob = zipped.read(archive._members[(reference_meta['identity'], component)])
            reference_points.append(_parse_swc_bytes(blob)[:, 2:5]*POSITION_UM)
    original = np.concatenate(reference_points)
    distance, _ = cKDTree(original).query(reference)
    report = dict(qualification='Coordinate and proximity diagnostics only; no anatomical attachment or completeness claim',
        swc_position_scale_um=POSITION_UM.tolist(), c3_position_scale_um=.001,
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        reference_skeleton=reference_meta, glia_skeleton=glia_meta,
        same_identity_frame_check=dict(source_vertices=len(original), c3_vertices=len(reference),
            nearest_distance_quantiles_um=dict(zip(['minimum', 'median', 'p95', 'maximum'],
                np.quantile(distance, [0, .5, .95, 1]).tolist())),
            coincident_fraction_1nm=float(np.mean(distance <= .001)),
            note='All proofread fragments retained for same-identity comparison; nearest vertices are not branch correspondences'),
        selected_neuron_proximity=sorted(proximity, key=lambda row: row['minimum_centerline_vertex_distance_um']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', newline='\n')
    print(json.dumps(dict(frame=report['same_identity_frame_check'], nearest=report['selected_neuron_proximity'][:3]), indent=2))


if __name__ == '__main__':
    main()
