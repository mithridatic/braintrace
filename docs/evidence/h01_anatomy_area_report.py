"""Audit the exact retained anatomy-transfer input and optional released mesh."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from h01_anatomy_area_audit import cable_geometry, mesh_geometry, read_swc


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def audit_case(archive, donor_manifest, *, mesh_prefix=None):
    """Compare source and converted geometry while requiring exact source hashes.

    Parameters
    ----------
    archive : pathlib.Path
        Original proofread SWC zip archive.
    donor_manifest : pathlib.Path
        Retained stage-1 donor.json beside its converted morphology.
    mesh_prefix : pathlib.Path or None, optional
        Artifact prefix from ``h01_anatomy_mesh_fetch``. Its cell ID, units,
        source URL and array hash are checked before reading the mesh.

    Returns
    -------
    dict
        Geometry measurements and provenance. A whole-cell surface is not
        equated with a partial converted tree, so a transfer verdict is withheld.
    """
    donor_bytes = donor_manifest.read_bytes()
    donor = json.loads(donor_bytes)
    member = donor['morphology_source']['member']
    cell_id = member.split('.')[0]
    converted_path = donor_manifest.parent / donor['morphology']
    converted_bytes = converted_path.read_bytes()
    with zipfile.ZipFile(archive) as source:
        raw_bytes = source.read(member)
        if _sha(raw_bytes) != donor['morphology_source']['member_sha256']:
            raise ValueError('Archive member differs from the registered transfer input.')
        members = sorted(name for name in source.namelist()
                         if name.startswith(cell_id + '.') and name.endswith('.swc'))
        components = {}
        for name in members:
            content = source.read(name)
            rows = read_swc(content.decode())
            components[name] = {
                'sha256': _sha(content),
                'geometry': cable_geometry(rows, position_um=(.032, .032, .033),
                                            radius_um=.001)}
    raw, converted = read_swc(raw_bytes.decode()), read_swc(converted_bytes.decode())
    result = {
        'cell_id': cell_id, 'selected_member': member,
        'donor_manifest_sha256': _sha(donor_bytes),
        'converted_swc_sha256': _sha(converted_bytes),
        'archive_sha256': _sha(archive.read_bytes()),
        'source_components': components,
        'selected_source_geometry': components[member]['geometry'],
        'converted_geometry': cable_geometry(
            converted, soma_sphere_radius_um=donor['conversion']['soma_radius_um']),
        'conversion_observations': {
            'source_nodes': len(raw), 'converted_nodes': len(converted),
            'source_soma_labeled_nodes': int((raw[:, 1] == 3).sum()),
            'source_dendrite_labeled_nodes': int((raw[:, 1] == 1).sum()),
            'source_nodes_below_radius_floor': int((raw[:, 5] * .001 < .05).sum()),
            'converted_nodes_at_radius_floor': int((converted[:, 5] == .05).sum()),
            'source_label_counts': {str(int(k)): int((raw[:, 1] == k).sum())
                                    for k in np.unique(raw[:, 1])},
            'converted_label_counts': {str(int(k)): int((converted[:, 1] == k).sum())
                                       for k in np.unique(converted[:, 1])},
        },
        'surface_mesh': None,
        'matched_membrane_area_ratio': None,
        'anatomy_transfer': 'unqualified',
        'interpretation': ('Source frusta, converted frusta and a separately reported donor soma '
                           'sphere are geometrical readings, not verified human membrane area. '
                           'No NEURON post-import or axon-replacement geometry is inferred.'),
        'missing_evidence': ['matched mesh component and anatomical-domain correspondence',
                             'surface-mesh area convergence',
                             'post-import simulation geometry',
                             'human response validation after transfer'],
    }
    if mesh_prefix is not None:
        array_bytes = mesh_prefix.with_suffix('.npz').read_bytes()
        record_bytes = mesh_prefix.with_suffix('.json').read_bytes()
        record = json.loads(record_bytes)
        if (record['cell_id'] != cell_id or record['array_sha256'] != _sha(array_bytes)
                or record['vertices_unit'] != 'nm' or record['source'] !=
                'https://h01-release.storage.googleapis.com/data/20210601/proofread_104'):
            raise ValueError('Mesh identity, units, source or artifact hash mismatch.')
        with np.load(mesh_prefix.with_suffix('.npz'), allow_pickle=False) as mesh:
            geometry = mesh_geometry(mesh['vertices'], mesh['faces'], position_um=(.001,)*3)
        result['surface_mesh'] = {'record_sha256': _sha(record_bytes),
                                  'array_sha256': _sha(array_bytes), 'lod': record['lod'],
                                  'geometry': geometry, 'domain': record['domain']}
    return result


def main(argv=None):
    """Write an immutable audit report for the retained transfer input.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments, or None to read process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--donor-manifest', required=True, type=Path)
    parser.add_argument('--mesh-prefix', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Preserve old evidence; choose a new output.')
    result = audit_case(args.archive, args.donor_manifest, mesh_prefix=args.mesh_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('cell_id', 'conversion_observations',
                                                  'matched_membrane_area_ratio')}))


if __name__ == '__main__':
    main()
