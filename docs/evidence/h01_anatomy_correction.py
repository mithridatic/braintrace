"""Verify the corrected export against original nodes, edges and annotations."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from h01_anatomy_area_audit import cable_geometry, read_swc


def verify_export(source, exported, conversion):
    """Check reversible identity and every physical coordinate, radius and edge.

    Parameters
    ----------
    source, exported : numpy.ndarray
        Original and corrected seven-column SWC rows.
    conversion : dict
        Sidecar containing source-to-export IDs and original annotation codes.

    Returns
    -------
    dict
        Verification results; mismatched topology, data or annotations raise.
    """
    original_geometry = cable_geometry(source, position_um=(.032, .032, .033), radius_um=.001)
    corrected_geometry = cable_geometry(exported)
    reverse = conversion['source_node_id_by_export_id']
    labels = conversion['source_annotation_by_node_id']
    original = {int(row[0]): row for row in source}
    exported_ids = {str(int(row[0])) for row in exported}
    if (len(exported) != len(source) or set(reverse) != exported_ids
            or set(reverse.values()) != set(original)
            or set(labels) != {str(k) for k in original}):
        raise ValueError('Source/export identifiers or annotation coverage differ.')
    position_errors, radius_errors = [], []
    for row in exported:
        source_id = reverse[str(int(row[0]))]
        reference = original[source_id]
        parent = -1 if row[6] == -1 else reverse[str(int(row[6]))]
        if parent != int(reference[6]) or labels[str(source_id)] != int(reference[1]) or row[1] != 0:
            raise ValueError('Topology, annotations or neutral export labels changed.')
        position_errors.append(float(np.abs(row[2:5] - reference[2:5] * (.032, .032, .033)).max()))
        radius_errors.append(float(abs(row[5] - reference[5] * .001)))
    if max(position_errors) > 1e-8 or max(radius_errors) > 1e-10:
        raise ValueError('Coordinate or radius changed beyond decimal serialization precision.')
    return {
        'status': 'passed', 'nodes_checked': len(source), 'edges_checked': len(source) - 1,
        'max_coordinate_error_um': max(position_errors), 'max_radius_error_um': max(radius_errors),
        'original_geometry': original_geometry, 'corrected_geometry': corrected_geometry,
        'annotation_code_counts': {str(int(k)): int((source[:, 1] == k).sum())
                                   for k in np.unique(source[:, 1])},
    }


def main(argv=None):
    """Write the corrective decision only after source-identity verification.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments, or None to read process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--historical-donor', type=Path, required=True)
    parser.add_argument('--corrected-anatomy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('A correction must use a new artifact path.')
    old = json.loads(args.historical_donor.read_bytes())
    new = json.loads(args.corrected_anatomy.read_bytes())
    source_meta = old['morphology_source']
    if source_meta != new['morphology_source']:
        raise ValueError('Historical and corrected exports have different sources.')
    with zipfile.ZipFile(args.archive) as archive:
        raw = archive.read(source_meta['member'])
    if hashlib.sha256(raw).hexdigest() != source_meta['member_sha256']:
        raise ValueError('Original archive member hash mismatch.')
    corrected_path = args.corrected_anatomy.parent / new['morphology']
    corrected_bytes = corrected_path.read_bytes()
    if hashlib.sha256(corrected_bytes).hexdigest() != new['morphology_sha256']:
        raise ValueError('Corrected SWC hash mismatch.')
    result = verify_export(read_swc(raw.decode()), read_swc(corrected_bytes.decode()), new['conversion'])
    historical_path = args.historical_donor.parent / old['morphology']
    historical_bytes = historical_path.read_bytes()
    result['historical_geometry'] = cable_geometry(read_swc(historical_bytes.decode()))
    result['cell_id'] = new['cell_id']
    result['source_member_sha256'] = source_meta['member_sha256']
    result['historical_swc_sha256'] = hashlib.sha256(historical_bytes).hexdigest()
    result['corrected_swc_sha256'] = hashlib.sha256(corrected_bytes).hexdigest()
    result['historical_reported_cable_um'] = sum(old['conversion'][k]
                                                 for k in ('apical_cable_um', 'basal_cable_um'))
    result['defect'] = 'H01 dendrite code 1 collapsed as SWC soma; H01 astrocyte code 2 relabeled axon'
    result['historical_transfer_interpretation'] = 'invalidated_by_conversion_defect'
    result['anatomy_transfer'] = 'unqualified; corrected physiology not measured'
    result['scope'] = 'Exact geometry preservation, not membrane equivalence or human physiology'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'nodes_checked', 'edges_checked',
                                             'historical_transfer_interpretation')}))


if __name__ == '__main__':
    main()
