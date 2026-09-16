"""Check full-population trace coverage against a verified construction anchor."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

_SIGNS = {'ei': {-1, 1}, 'e_only': {1}, 'i_only': {-1}, 'disconnected': set()}
_MODEL_FIELDS = ('simulated_cell_ids', 'cells', 'donors', 'compartments_by_cell', 'n_compartments')


def audit_runtime(build, reference, plan, arrays):
    """Verify bounded runtime without promoting biological qualification.

    Parameters
    ----------
    build, reference : dict
        Run metadata and independently verified full construction metadata.
    plan : dict
        Registered cell count, control, dt and duration.
    arrays : mapping
        Saved trace arrays, including end-of-step times.

    Returns
    -------
    dict
        Failures and per-cell observations for this runtime window only.
    """
    failures, observations = [], []

    def require(condition, message):
        if not condition:
            failures.append(message)

    identities = reference.get('simulated_cell_ids', [])
    require(len(identities) > 0 and len(identities) == len(set(identities)) == plan.get('cells'),
            'reference and plan must cover exactly the same unique cells')
    for key in _MODEL_FIELDS:
        require(build.get(key) == reference.get(key), 'model metadata differs: '+key)
    control = plan.get('control')
    signs = _SIGNS.get(control)
    require(signs is not None and build.get('control') == control, 'control differs or is invalid')
    expected_contacts = [dict(edge, enabled=edge['dale_sign'] in (signs or set()))
                         for edge in reference.get('contacts', [])]
    require(build.get('contacts') == expected_contacts, 'contact settings differ')
    for field, enabled in [('enabled_contacts', True), ('removed_contacts', False)]:
        expected = [edge['annotation_id'] for edge in expected_contacts if edge['enabled'] == enabled]
        require(build.get(field) == expected, field+' differs')
    dt, duration = plan.get('dt_ms'), plan.get('duration_ms')
    valid_time = (isinstance(dt, (float, int)) and isinstance(duration, (float, int))
                  and np.isfinite([dt, duration]).all() and dt > 0 and duration > 0)
    require(valid_time, 'invalid planned dt or duration')
    steps = int(round(duration/dt)) if valid_time else 0
    require(valid_time and steps > 0 and np.isclose(steps*dt, duration, rtol=0, atol=1e-12),
            'duration is not a positive whole number of steps')
    require(build.get('dt_ms') == dt and build.get('duration_ms') == duration, 'run timing metadata differs')
    times = np.asarray(arrays.get('time_ms', []))
    grid_ok = (valid_time and times.shape == (steps,) and times.dtype.kind in 'fi' and
               np.allclose(times, np.arange(1, steps+1)*dt, rtol=0, atol=1e-12))
    require(grid_ok, 'time grid differs from end-of-step convention')
    keys = list(arrays)
    require(len(keys) == len(set(keys)), 'duplicate array names')
    for name in keys:
        value = np.asarray(arrays[name])
        require(value.dtype.kind in 'biuf' and np.isfinite(value).all(), 'nonfinite or nonnumeric array: '+name)
        if name != 'time_ms':
            parts = name.split('_', 2)
            require(len(parts) == 3 and parts[0] == 'cell' and parts[1] in identities,
                    'array belongs to an unexpected population: '+name)
            require(value.shape == (steps, 1), 'array shape differs: '+name)
    for identity in identities:
        prefix = 'cell_'+identity
        names = [prefix+suffix for suffix in ('_voltage', '_output_voltage', '_events')]
        present = all(name in arrays for name in names)
        require(present, identity+': missing voltage/output/event array')
        if not present:
            continue
        voltage, output, events = (np.asarray(arrays[name]) for name in names)
        require(voltage.dtype.kind == output.dtype.kind == 'f', identity+': voltage arrays must be floating point')
        require(events.dtype.kind == 'b', identity+': events must be Boolean')
        if (grid_ok and voltage.shape == output.shape == events.shape == (steps, 1) and voltage.size
                and voltage.dtype.kind == output.dtype.kind == 'f' and events.dtype.kind == 'b'
                and all(np.isfinite(value).all() for value in (voltage, output, events))):
            observations.append(dict(cell_id=identity, voltage_min_mv=float(voltage.min()),
                voltage_max_mv=float(voltage.max()), spike_count=int(events.sum()),
                spike_times_ms=times[events[:, 0].astype(bool)].tolist()))
    return dict(status='passed' if not failures else 'failed', scope='finite runtime window only',
                cells=len(identities), dt_ms=dt, duration_ms=duration, steps=steps, control=control,
                failures=failures, observations=observations,
                activity_qualified=False, physiology_qualified=False)


def main():
    """Write a runtime verdict linked to raw artifacts by SHA-256.

    Returns
    -------
    None
        Saves the decision and exits nonzero if it fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'reference', 'construction-decision', 'plan', 'traces', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    paths = {key:getattr(args, key) for key in ('build', 'reference', 'construction_decision', 'plan', 'traces')}
    payloads = {key:path.read_bytes() for key,path in paths.items()}
    hashes = {key:hashlib.sha256(value).hexdigest() for key,value in payloads.items()}
    decision = json.loads(payloads['construction_decision'])
    if decision.get('status') != 'passed' or decision['inputs']['build']['sha256'] != hashes['reference']:
        raise ValueError('Construction reference is not the gate-verified artifact.')
    with np.load(io.BytesIO(payloads['traces']), allow_pickle=False) as arrays:
        result = audit_runtime(json.loads(payloads['build']), json.loads(payloads['reference']),
                               json.loads(payloads['plan']), arrays)
    result['inputs'] = {key:dict(path=str(path), sha256=hashes[key]) for key,path in paths.items()}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status', 'cells', 'steps', 'control', 'failures')}), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
