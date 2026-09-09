"""Compare complete H01 populations at a registered timestep halving."""
import numpy as np
from docs.evidence.h01_population_runtime_gate import audit_runtime


def audit_refinement(reference, coarse, fine):
    """Check voltage and event convergence for all104 cells.

    Parameters
    ----------
    reference : dict
        Construction-qualified reference metadata.
    coarse, fine : tuple
        Build metadata, registered plan and saved arrays for each run. Callers
        must hash-bind these inputs when persisting a verdict.

    Returns
    -------
    dict
        Numerical comparison only; physiological and activity gates stay open.
    """
    failures, observations = [], []
    def result():
        return dict(status='failed' if failures else 'passed',failures=failures,
            observations=observations,scope='full104 voltage and event timestep comparison only',
            voltage_allowance_mV=1.,event_allowance_ms=.05,
            activity_qualified=False,physiology_qualified=False)
    for name,run in [('coarse',coarse),('fine',fine)]:
        verdict=audit_runtime(run[0],reference,run[1],run[2])
        failures.extend(name+': '+failure for failure in verdict['failures'])
    if failures:
        return result()
    cp,fp=coarse[1],fine[1]
    if cp['duration_ms']!=fp['duration_ms'] or cp['control']!=fp['control']:
        failures.append('Duration or control differs.')
    if cp['dt_ms']!=2*fp['dt_ms']:
        failures.append('Fine timestep is not half the coarse timestep.')
    if failures:
        return result()
    ca,fa=coarse[2],fine[2]
    if not np.allclose(ca['time_ms'],fa['time_ms'][1::2],rtol=0,atol=1e-12):
        failures.append('Matched end-step times differ.')
        return result()
    for identity in reference['simulated_cell_ids']:
        prefix='cell_'+identity
        errors={site:float(np.max(np.abs(ca[prefix+'_'+site]-fa[prefix+'_'+site][1::2])))
                for site in ('voltage','output_voltage')}
        old=ca['time_ms'][ca[prefix+'_events'][:,0]]
        new=fa['time_ms'][fa[prefix+'_events'][:,0]]
        same=len(old)==len(new)
        timing=float(np.max(np.abs(old-new))) if same and len(old) else (0. if same else None)
        passed=max(errors.values())<=1. and same and timing<=.05
        observations.append(dict(cell_id=identity,voltage_errors_mV=errors,
            coarse_events=len(old),fine_events=len(new),max_event_timing_error_ms=timing,passed=passed))
        if not passed:
            failures.append(identity+': voltage or event refinement exceeds allowance')
    return result()
