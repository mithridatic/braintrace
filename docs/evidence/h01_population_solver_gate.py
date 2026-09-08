"""Check that a full104 construction changed only its numerical solver."""
import argparse
import hashlib
import json
from pathlib import Path


def audit_solver_transition(previous, current):
    """Compare the complete model metadata across an explicit solver change.

    Parameters
    ----------
    previous, current : dict
        Old scan-solver and new implicit-solver construction evidence.

    Returns
    -------
    dict
        Metadata verdict; source, runtime and physiology gates remain separate.
    """
    failures = []
    ids = previous.get('simulated_cell_ids', [])
    if len(ids) != 104 or len(set(ids)) != 104:
        failures.append('Reference must contain exactly 104 unique identities.')
    for key in ('simulated_cell_ids','donors','compartments_by_cell','n_compartments',
                'control','contacts','enabled_contacts','removed_contacts'):
        if key not in previous or key not in current or previous[key] != current[key]:
            failures.append('Model metadata differs or is missing: '+key)
    old_cells, new_cells = previous.get('cells',{}), current.get('cells',{})
    if set(old_cells) != set(ids) or set(new_cells) != set(ids):
        failures.append('Per-cell identity set differs.')
    for identity in ids:
        old, new = dict(old_cells.get(identity,{})), dict(new_cells.get(identity,{}))
        if old.pop('solver',None) != 'h01_staggered_scan':
            failures.append(identity+': unexpected original solver')
        if new.pop('solver',None) != 'h01_staggered_calcium_implicit':
            failures.append(identity+': unexpected replacement solver')
        if old != new:
            failures.append(identity+': model fields other than solver differ')
    return dict(status='failed' if failures else 'passed',failures=failures,
                scope='solver transition metadata only',cells=len(ids),
                construction_qualified=False,runtime_qualified=False,physiology_qualified=False)


def main():
    """Write a hash-bound verdict for two construction artifacts.

    Returns
    -------
    None
        Writes JSON and exits nonzero when the comparison fails.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','current','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    payloads={key:getattr(args,key).read_bytes() for key in ('previous','current')}
    result=audit_solver_transition(*(json.loads(payloads[key]) for key in ('previous','current')))
    result['inputs']={key:dict(path=str(getattr(args,key)),sha256=hashlib.sha256(value).hexdigest())
                      for key,value in payloads.items()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
