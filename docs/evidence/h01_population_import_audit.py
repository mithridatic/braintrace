"""Audit all selected H01 components against the normalized source segments."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import brainunit as u
import numpy as np

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_network_init import heartbeat


def segment_counter(proximal, distal):
    """Return the multiset of directed physical segments.

    Parameters
    ----------
    proximal, distal : array_like
        Segment endpoint coordinates in micrometers.

    Returns
    -------
    Counter
        Directed endpoints rounded to a nanometer-millionth for serialization.
    """
    return Counter(tuple(row) for row in np.round(np.column_stack((proximal, distal)), 9))


def audit_geometry(component):
    """Compare imported segments with source topology and physical coordinates.

    Parameters
    ----------
    component : H01Component
        Loaded component with its normalized source.

    Returns
    -------
    dict
        Counts of missing and extra directed segments, and the pass verdict.
    """
    rows = np.loadtxt(component.normalized_swc.splitlines())
    lookup = {int(row[0]): row[2:5] for row in rows}
    children = rows[rows[:, 6] != -1]
    expected = segment_counter([lookup[int(p)] for p in children[:, 6]], children[:, 2:5])
    branches = component.morphology.branches
    observed = segment_counter(
        np.concatenate([b.points_proximal.to_decimal(u.um) for b in branches]),
        np.concatenate([b.points_distal.to_decimal(u.um) for b in branches]),
    )
    missing, extra = sum((expected-observed).values()), sum((observed-expected).values())
    return dict(source_segments=sum(expected.values()), imported_segments=sum(observed.values()),
                missing_segments=missing, extra_segments=extra, passed=missing == extra == 0)


def audit_soma(component, polarity):
    """Check the measured soma against the electrical partition.

    Parameters
    ----------
    component : H01Component
        Imported source component.
    polarity : str
        E or I from the prepared topology's Dale sign.

    Returns
    -------
    dict
        Selected soma location and strict electrical-region membership.
    """
    from braintrace.datasets.h01_network import _regions
    from braintrace.datasets.h01_ei_cell import _validate_regions

    intervals = _validate_regions(component.morphology, _regions(component), polarity)
    points = component.anatomy().soma_location().evaluate(component.morphology).points
    inside = all(any(branch == b and lo <= x <= hi for b, lo, hi in intervals['soma'])
                 for branch, x in points)
    return dict(soma_location=points, electrical_regions_valid=True, soma_inside_region=inside)


def selected_component(row, soma_bearing):
    """The inventory row's largest component, or its largest soma-bearing one when asked and it differs.

    Parameters
    ----------
    row : dict
        One ``cells`` entry of ``h01-population-components.json``.
    soma_bearing : bool
        Prefer the first (largest) entry of ``soma_components`` when the largest lacks a soma.

    Returns
    -------
    int
        Component suffix to load.
    """
    if soma_bearing and not row.get('largest_has_soma', True) and row.get('soma_components'):
        return row['soma_components'][0]
    return row['largest_component']


def main():
    """Write a checkpointed import audit.

    Returns
    -------
    None
        Writes per-cell evidence and prints progress.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--topology', type=Path, help='Also check soma/region membership using these cell polarities.')
    parser.add_argument('--expected-sha256', help='Archive digest other than the pinned proofread release.')
    parser.add_argument('--source', help='Source label recorded for a non-proofread archive.')
    parser.add_argument('--soma-bearing', action='store_true',
                        help='Audit the largest soma-bearing component when the largest component lacks a soma.')
    args = parser.parse_args()
    archive = H01Archive(args.archive, expected_sha256=args.expected_sha256, source=args.source)
    inventory = json.loads(args.components.read_text())
    polarities = None
    if args.topology:
        polarities = {n['cell_id']: {1:'E', -1:'I'}[n['dale_sign']]
                      for n in json.loads(args.topology.read_text())['nodes']}
        if set(polarities) != {c['cell_id'] for c in inventory['cells']}:
            raise ValueError('Topology and component inventory populations differ.')
    result = dict(status='running', archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  source=archive.source, cells=[],
                  scope='largest soma-bearing component of each anatomical identity')
    expected = len(inventory['cells'])
    started = time.perf_counter()
    for row in inventory['cells']:
        selected = selected_component(row, args.soma_bearing)
        record = dict(cell_id=row['cell_id'], component=selected)
        tick = time.perf_counter()
        try:
            with heartbeat('import '+row['cell_id'], lambda message: print(message, flush=True), seconds=30):
                component = archive.load(row['cell_id'], component=selected)
            record.update(audit_geometry(component), source_sha256=component.source_sha256)
            if polarities is not None:
                record.update(audit_soma(component, polarities[row['cell_id']]))
                record['passed'] = record['passed'] and record['soma_inside_region']
        except Exception as exc:
            record.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        record['seconds'] = time.perf_counter()-tick
        result['cells'].append(record)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(f"{len(result['cells'])}/{expected} {record}", flush=True)
    result.update(status='completed', seconds=time.perf_counter()-started,
                  passed=len(result['cells']) == expected and all(r['passed'] for r in result['cells']))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
