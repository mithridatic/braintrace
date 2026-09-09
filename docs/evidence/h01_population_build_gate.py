"""Verify full-population construction identity and provenance against sources."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def audit_build(build, imports, topology):
    """Check a population build against independent source evidence.

    Parameters
    ----------
    build : dict
        Raw network construction evidence.
    imports : dict
        Completed full-population import audit.
    topology : dict
        Prepared anatomical topology with supported-contact flags.

    Returns
    -------
    dict
        Construction verdict and explicit failures. No runtime or physiology
        qualification is inferred.
    """
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    source_rows = imports.get('cells', [])
    source_ids = [row['cell_id'] for row in source_rows]
    expected = set(source_ids)
    require(imports.get('status') == 'completed' and imports.get('passed') is True,
            'import audit is not a completed pass')
    require(len(source_ids) == len(expected) == 104, 'import audit must contain 104 unique cells')
    require(all(row.get('passed') is True for row in source_rows), 'a source component failed import')
    require(_sha256(imports.get('archive_sha256')), 'source archive hash is missing or malformed')
    topology_ids = [row['cell_id'] for row in topology.get('nodes', [])]
    require(len(topology_ids) == len(set(topology_ids)) and set(topology_ids) == expected,
            'topology population differs from import population')
    simulated = build.get('simulated_cell_ids', [])
    require(len(simulated) == len(set(simulated)) and set(simulated) == expected,
            'simulated population is incomplete, duplicated or substituted')
    cells, counts = build.get('cells', {}), build.get('compartments_by_cell', {})
    require(set(cells) == expected, 'per-cell evidence population differs')
    require(set(counts) == expected, 'compartment-count population differs')
    require(set(build.get('donors', {})) == expected, 'donor population differs')
    require(topology.get('archive_sha256') == imports.get('archive_sha256'), 'topology archive hash differs')
    valid_counts = all(type(n) is int and n > 0 for n in counts.values())
    require(valid_counts, 'compartment counts must be positive integers')
    require(valid_counts and build.get('n_compartments') == sum(counts.values()),
            'total compartment count differs')
    for source in source_rows:
        identity = source['cell_id']
        record = cells.get(identity, {})
        provenance = record.get('measured_anatomy', {})
        require(provenance.get('member') == f"{identity}.{source['component']}.swc",
                identity+': component differs')
        require(_sha256(source.get('source_sha256'))
                and provenance.get('source_sha256') == source.get('source_sha256'),
                identity+': source hash differs')
        require(provenance.get('archive_sha256') == imports.get('archive_sha256'),
                identity+': archive hash differs')
        require(record.get('n_compartments') == counts.get(identity) and identity in counts,
                identity+': per-cell compartment count differs')
        require(bool(record.get('donor')) and record.get('donor') == build.get('donors', {}).get(identity),
                identity+': donor evidence differs')
    supported = {edge['annotation_id']: edge for edge in topology.get('contacts', [])
                 if edge.get('construction_ready') is True}
    contacts = build.get('contacts', [])
    contact_ids = [edge['annotation_id'] for edge in contacts]
    require(len(contact_ids) == len(set(contact_ids)) and set(contact_ids) == set(supported),
            'contacts differ from supported anatomical contacts')
    require(build.get('control') == 'ei', 'build gate requires the complete ei condition')
    require(build.get('enabled_contacts') == contact_ids and not build.get('removed_contacts'),
            'enabled-contact evidence differs')
    for edge in contacts:
        original = supported.get(edge['annotation_id'], {})
        require(edge.get('enabled') is True, edge['annotation_id']+': supported contact disabled')
        require(all(edge.get(k) == original.get(k) for k in ('pre_cell', 'post_cell', 'dale_sign')),
                edge['annotation_id']+': contact endpoints or sign differ')
    return dict(status='passed' if not failures else 'failed', scope='104-cell construction only',
                expected_cells=len(expected), simulated_cells=len(simulated),
                n_compartments=build.get('n_compartments'), failures=failures,
                runtime_qualified=False, physiology_qualified=False)


def main():
    """Write a hash-linked construction verdict and fail on rejected evidence.

    Returns
    -------
    None
        Writes JSON evidence; exits nonzero if construction is not verified.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--imports', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in ('build', 'imports', 'topology')}
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    result = audit_build(*(json.loads(payloads[key]) for key in ('build', 'imports', 'topology')))
    result['inputs'] = {key: dict(path=str(paths[key]), sha256=hashlib.sha256(value).hexdigest())
                        for key, value in payloads.items()}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
