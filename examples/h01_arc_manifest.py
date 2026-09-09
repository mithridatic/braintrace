"""Export a content-addressed H01 lifecycle manifest from verified population evidence."""

import argparse
import json
import os
from pathlib import Path

import brainstate

from braintrace.datasets.h01 import H01Archive
from examples.pp_prop.h01_checkpoint import store_asset
from examples.pp_prop.h01_runtime import topology_from_evidence
from examples.pp_prop.h01_session import numerical_settings


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
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text())
    evidence = report['evidence']
    if (report.get('status') not in ('forward_pass', 'learning_probe_pass') or
            len(evidence['simulated_cell_ids']) != 104 or
            report.get('nonfinite_states') != []):
        raise ValueError('Initial H01 lifecycle manifest requires successful 104-cell population evidence')
    contacts = json.loads(args.contacts.read_text())
    asset_root = args.output.parent/'assets'
    digest = store_asset(args.archive, asset_root)
    with brainstate.environ.context(precision=64):
        topology = topology_from_evidence(evidence, contacts, H01Archive(asset_root/digest))
        document = dict(schema='h01-arc-manifest-v1', topology=topology.to_dict(),
            settings=numerical_settings(), assets=[digest],
            asset_root=os.path.relpath(asset_root, args.output.parent))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(manifest=str(args.output.resolve()), original_cells=104,
                         anatomical_contacts=len(topology.to_dict()['active_contacts']), synthetic_contacts=0)))


if __name__ == '__main__':
    main()
