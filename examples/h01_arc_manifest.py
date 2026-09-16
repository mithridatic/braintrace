"""Export a content-addressed H01 lifecycle manifest from verified population evidence.

Every ``--archive`` is stored as an immutable asset and listed in ``assets``; a cell resolves
to its archive through the ``archive_sha256`` its construction evidence recorded. Further
probe evidence (``--extra-evidence``) is merged before any kept subset is taken.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import brainstate

from examples.pp_prop.h01_checkpoint import store_asset
from examples.pp_prop.h01_runtime import open_archives, topology_from_evidence
from examples.pp_prop.h01_session import numerical_settings


SOURCE_CELLS = 104


def kept_subset(evidence, kept):
    """Restrict population evidence to the kept cells and the contacts between them.

    Parameters
    ----------
    evidence : dict
        Successful source network construction evidence.
    kept : list of str
        Cell identities to keep; every one must be a simulated cell.

    Returns
    -------
    dict
        Copy of ``evidence`` with ``simulated_cell_ids``, ``cells`` and ``contacts``
        filtered; every other key is carried through unchanged.
    """
    wanted = set(kept)
    missing = sorted(wanted-set(evidence['simulated_cell_ids']))
    if not wanted:
        raise ValueError('A kept-cell list must name at least one cell')
    if missing:
        raise ValueError(f'Kept cells are not simulated cells: {", ".join(missing)}')
    filtered = dict(evidence)
    filtered['simulated_cell_ids'] = [c for c in evidence['simulated_cell_ids'] if c in wanted]
    filtered['cells'] = {c: r for c, r in evidence.get('cells', {}).items() if c in wanted}
    filtered['contacts'] = [e for e in evidence.get('contacts', [])
                            if e['pre_cell'] in wanted and e['post_cell'] in wanted]
    return filtered


def merge_evidence(primary, extras):
    """Merge further probe evidence (another archive's cells) into one population.

    Parameters
    ----------
    primary : dict
        Successful source network construction evidence.
    extras : sequence of dict
        Further construction evidence; cell identities must not repeat.

    Returns
    -------
    dict
        Copy of ``primary`` with ``simulated_cell_ids``, ``cells`` and ``contacts``
        extended in order; every other key is carried through from ``primary``.
    """
    merged = dict(primary)
    merged['simulated_cell_ids'] = list(primary['simulated_cell_ids'])
    merged['cells'] = dict(primary.get('cells', {}))
    merged['contacts'] = list(primary.get('contacts', []))
    for extra in extras:
        repeated = sorted(set(extra['simulated_cell_ids']) & set(merged['simulated_cell_ids']))
        if repeated:
            raise ValueError(f'Extra evidence repeats simulated cells: {", ".join(repeated)}')
        merged['simulated_cell_ids'] += list(extra['simulated_cell_ids'])
        merged['cells'].update(extra.get('cells', {}))
        merged['contacts'] += list(extra.get('contacts', []))
    return merged


def _successful_report(path, expected=None):
    """Load a probe report; it must record a successful, finite construction."""
    report = json.loads(Path(path).read_text())
    count = len(report.get('evidence', {}).get('simulated_cell_ids', []))
    if (report.get('status') not in ('forward_pass', 'learning_probe_pass') or
            report.get('nonfinite_states') != [] or (expected is not None and count != expected)):
        wanted = f'{expected}-cell ' if expected is not None else ''
        raise ValueError(f'Initial H01 lifecycle manifest requires successful {wanted}population evidence')
    return report


def _kept_cells(paths):
    """Join the kept lists of one or more keep/drop decisions, with provenance rows."""
    kept, provenance = [], []
    for path in paths:
        kept += [c for c in json.loads(path.read_text())['kept'] if c not in kept]
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return kept, provenance


def main():
    """Write the pinned source manifest consumed by the H01 evolution backend.

    Returns
    -------
    None
        Writes a manifest and its immutable source morphology assets.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/h01-arc-probe-104.json'))
    parser.add_argument('--extra-evidence', type=Path, action='append', default=[],
                        help='Further probe evidence merged before the kept subset (repeatable).')
    parser.add_argument('--contacts', type=Path, default=Path('docs/evidence/h01-verified-network.json'))
    parser.add_argument('--archive', type=Path, action='append', required=True,
                        help='Source morphology archive; repeat for every archive the cells come from.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=Path, action='append', default=None,
                        help="Keep/drop decision JSON whose top-level 'kept' list restricts the manifest "
                             "(repeatable; the kept lists are joined).")
    args = parser.parse_args()
    whole = args.cells is None and not args.extra_evidence
    report = _successful_report(args.evidence, SOURCE_CELLS if whole else None)
    extras = [_successful_report(path)['evidence'] for path in args.extra_evidence]
    evidence = merge_evidence(report['evidence'], extras)
    source_cells = len(evidence['simulated_cell_ids'])
    kept_from = None
    if args.cells is not None:
        kept, provenance = _kept_cells(args.cells)
        evidence = kept_subset(evidence, kept)
        kept_from = dict(provenance[0], source_cells=source_cells)
        if len(provenance) > 1:
            kept_from['additional'] = provenance[1:]
    expected = len(evidence['simulated_cell_ids'])
    contacts = json.loads(args.contacts.read_text())
    asset_root = args.output.parent/'assets'
    digests = list(dict.fromkeys(store_asset(path, asset_root) for path in args.archive))
    with brainstate.environ.context(precision=64):
        topology = topology_from_evidence(evidence, contacts, open_archives(asset_root, digests))
        document = dict(schema='h01-arc-manifest-v1', topology=topology.to_dict(),
            settings=numerical_settings(), assets=digests,
            asset_root=os.path.relpath(asset_root, args.output.parent), original_cells=expected)
        if kept_from is not None:
            document['kept_from'] = kept_from
        if extras:
            document['extra_evidence'] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                          for path in args.extra_evidence]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(manifest=str(args.output.resolve()), original_cells=expected,
                         anatomical_contacts=len(topology.to_dict()['active_contacts']), synthetic_contacts=0)))


if __name__ == '__main__':
    main()
