"""Validate delivery and isolation in the four disjoint-pair H01 controls."""

import numpy as np

from docs.evidence.h01_population_runtime_gate import audit_runtime

CONTROLS = ('ei', 'e_only', 'i_only', 'disconnected')
ATOL_MV = 1e-5  # GPU atomic_add in the DHS kernel leaves ~9e-7 mV run-to-run differences


def audit_controls(reference, runs):
    """Check matched controls without promoting physiological qualification.

    Parameters
    ----------
    reference : dict
        Independently construction-qualified full-population metadata.
    runs : dict
        Each control maps to ``(build_metadata, registered_plan, arrays)``.
        Callers must hash-bind these inputs when saving the returned decision.

    Returns
    -------
    dict
        Delivery observations and failures, scoped to two disjoint pairs.
    """
    failures, observations, largest = [], [], [0.]

    def result():
        return dict(status='failed' if failures else 'passed', failures=failures,
                    scope='delivery and control isolation only', observations=observations,
                    physiology_qualified=False, functional_inhibition_qualified=False,
                    tolerance_mv=ATOL_MV, max_abs_difference_mv=largest[0])

    if set(runs) != set(CONTROLS):
        failures.append('exactly four named controls are required')
        return result()
    edges = reference.get('contacts', [])
    endpoints = [edge[side+'_cell'] for edge in edges for side in ('pre', 'post')]
    if len(edges) != 2 or len(set(endpoints)) != 4:
        failures.append('checker requires exactly two disjoint directed pairs')
        return result()
    for control, (build, plan, arrays) in runs.items():
        checked = audit_runtime(build, reference, plan, arrays)
        if plan.get('control') != control or checked['status'] != 'passed':
            failures.append(control+': invalid individual runtime evidence')
    if failures:
        return result()
    baseline = runs['ei'][2]
    for control, (build, plan, arrays) in runs.items():
        if (set(arrays) != set(baseline) or
                not np.array_equal(arrays['time_ms'], baseline['time_ms'])):
            failures.append(control+': trace keys or time grid differ')
            continue
        receivers = {edge['post_cell'] for edge in build['contacts'] if not edge['enabled']}
        for name, values in arrays.items():
            if name != 'time_ms' and name.split('_', 2)[1] not in receivers:
                difference = np.abs(np.asarray(values, float)-np.asarray(baseline[name], float))
                largest[0] = max(largest[0], float(np.nanmax(difference)) if difference.size else 0.)
                if not np.allclose(values, baseline[name], rtol=0., atol=ATOL_MV):
                    failures.append(control+': change outside removed receivers: '+name)
        times = np.asarray(arrays['time_ms'])
        dt = plan['dt_ms']
        for edge in build['contacts']:
            label = control+'/'+edge['annotation_id']
            prefix = 'cell_'+edge['post_cell']+'_syn_'+edge['annotation_id']
            if prefix+'_g' not in arrays or prefix+'_voltage' not in arrays:
                failures.append(label+': missing receptor probes')
                continue
            conductance = np.asarray(arrays[prefix+'_g'])[:, 0]
            if np.any(conductance < 0):
                failures.append(label+': negative conductance')
            if not edge['enabled']:
                if np.any(conductance != 0):
                    failures.append(label+': disabled receptor has conductance')
                continue
            events = np.asarray(arrays['cell_'+edge['pre_cell']+'_events'])[:, 0]
            arrivals = times[events]+edge['delay_ms']
            observable = arrivals[arrivals <= times[-1]-2*dt]
            rises = times[np.diff(conductance, prepend=0.) > 0]
            if not observable.size:
                failures.append(label+': no source event with an observable delivery window')
            allowance = 2*dt+1e-12
            if any(not np.any(np.abs(rises-arrival) <= allowance) for arrival in observable):
                failures.append(label+': source event lacks a conductance rise')
            if any(not np.any(np.abs(arrivals-rise) <= allowance) for rise in rises):
                failures.append(label+': conductance rise lacks a delayed source event')
            observations.append(dict(control=control, annotation_id=edge['annotation_id'],
                observable_source_events=int(observable.size), conductance_rises=int(rises.size),
                peak_conductance_us=float(conductance.max()), arrival_allowance_ms=2*dt))
    return result()
