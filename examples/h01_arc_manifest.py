"""Export a content-addressed H01 lifecycle manifest from verified population evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import brainstate

from braintrace.datasets.h01 import H01Archive
from examples.pp_prop.h01_checkpoint import store_asset
from examples.pp_prop.h01_runtime import topology_from_evidence
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


def main():
    """Write the pinned source manifest consumed by the H01 evolution backend.

    Returns
    -------
    None
        Writes a manifest and its immutable source morphology asset.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence/h01-arc-probe-104.json'))
    parser.add_argument('--contacts', type=Path, default=Path('docs/evidence/h01-verified-network.json'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=Path,
                        help="Keep/drop decision JSON whose top-level 'kept' list restricts the manifest.")
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text())
    evidence = report['evidence']
    kept_from = None
    if args.cells is not None:
        kept = json.loads(args.cells.read_text())['kept']
        evidence = kept_subset(evidence, kept)
        kept_from = dict(path=str(args.cells), sha256=hashlib.sha256(args.cells.read_bytes()).hexdigest(),
                         source_cells=SOURCE_CELLS)
    expected = SOURCE_CELLS if kept_from is None else len(kept)
    if (report.get('status') not in ('forward_pass', 'learning_probe_pass') or
            len(evidence['simulated_cell_ids']) != expected or
            report.get('nonfinite_states') != []):
        raise ValueError(f'Initial H01 lifecycle manifest requires successful {expected}-cell population evidence')
    contacts = json.loads(args.contacts.read_text())
    asset_root = args.output.parent/'assets'
    digest = store_asset(args.archive, asset_root)
    with brainstate.environ.context(precision=64):
        topology = topology_from_evidence(evidence, contacts, H01Archive(asset_root/digest))
        document = dict(schema='h01-arc-manifest-v1', topology=topology.to_dict(),
            settings=numerical_settings(), assets=[digest],
            asset_root=os.path.relpath(asset_root, args.output.parent), original_cells=expected)
        if kept_from is not None:
            document['kept_from'] = kept_from
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(manifest=str(args.output.resolve()), original_cells=expected,
                         anatomical_contacts=len(topology.to_dict()['active_contacts']), synthetic_contacts=0)))


if __name__ == '__main__':
    main()
