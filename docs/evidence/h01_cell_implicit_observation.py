"""Test the experimental calcium solver on the exact isolated H01 failure."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import brainstate
import brainunit as u
import numpy as np
from braincell.filter import AtLocation
from braincell.mech import CurrentProbe, MechanismProbe

from braintrace.datasets import h01_calcium_solver
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_network import make_h01_network
from braintrace.datasets.h01_network_init import init_h01_network_states, heartbeat, process_rss_mb

def require_same_metadata(actual, reference):
    """Require equality after converting in-memory tuples to JSON arrays.

    Parameters
    ----------
    actual, reference : dict
        Live cell metadata and the persisted reference record.

    Returns
    -------
    None
        Raises AssertionError when any serialized value differs.
    """
    assert json.loads(json.dumps(actual)) == reference, 'Cell metadata changed before observation.'


def main():
    """Run the registered unchanged-cell observation and save its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt-ms', type=float, choices=[.005,.0025,.00125,.000625,.0003125], default=.005)
    args = parser.parse_args()
    started = time.perf_counter()
    emit = lambda message: print(f'[{time.perf_counter()-started:.1f}s] {message}', flush=True)
    identity = '7196644737'
    reference_path = Path('.cache/h01/readiness/104-build-r2-build.json')
    reference = json.loads(reference_path.read_text())['cells'][identity]
    baseline_path = Path('.cache/h01/readiness/cell7196644737-'+('dt005' if args.dt_ms == .005 else 'dt0025')+'-traces.npz')
    topology = json.loads(Path('docs/evidence/h01-ready-cell7196644737-topology.json').read_text())
    components = json.loads(Path('docs/evidence/h01-population-components.json').read_text())
    archive = H01Archive(Path('.cache/h01/proofread104.zip'))
    geometry = lambda cell: [[int(cv.id), int(cv.branch_id) if cv.branch_id is not None else None, float(cv.prox), float(cv.dist),
                             float(cv.area.to_decimal(u.um**2))] for cv in cell.cvs]
    with brainstate.environ.context(precision=64):
        network, evidence = make_h01_network(topology, archive, H01Annotations(Path('.cache/h01')),
            include_isolated=True, cells=1, components=components, currents_na={identity:1.}, progress=emit, solver='h01_staggered_calcium_implicit')
        expected = dict(reference, solver='h01_staggered_calcium_implicit')
        require_same_metadata(evidence['cells'][identity], expected)
        cell = network.populations['cell_'+identity].cell
        before = geometry(cell)
        location = AtLocation(*reference['output_site']['output_midpoint'])
        probes = [('calcium', 'Ci', 'calcium_mM'), ('calcium', 'E', 'calcium_reversal_mV'),
                  ('pv_SK', 'z', 'sk_z'), ('pv_NaTs', 'm', 'nat_m'), ('pv_NaTs', 'h', 'nat_h'),
                  ('pv_Kv3_1', 'm', 'kv3_m')]
        for mechanism, field, name in probes:
            cell.place(location, MechanismProbe(mechanism=mechanism, field=field, name=name))
        cell.place(location, CurrentProbe(ion='calcium', name='ica_inward_ma_cm2'))
        for mechanism, name in [('pv_Kv3_1','ikv3_inward_ma_cm2'), ('pv_NaTs','ina_inward_ma_cm2')]:
            cell.place(location, CurrentProbe(mechanism=mechanism, name=name))
        assert geometry(cell) == before, 'Probe placement changed compartment geometry.'
        soma = reference['electrical_intervals']['soma']
        soma_area = sum(area for _,branch,lo,hi,area in before
                        if any(b == branch and start <= (lo+hi)/2 <= end for b,start,end in soma))
        cv_id = reference['output_site']['output_cv_id']
        output_area = next(row[4] for row in before if row[0] == cv_id)
        emit(f'Output CV area {output_area:.9g} um2; modeled soma area {soma_area:.9g} um2')
        init_h01_network_states(network, progress=emit, heartbeat_seconds=30.)
        with heartbeat('state observation', emit, seconds=30.):
            result = network.run(dt=args.dt_ms*u.ms, duration=10.*u.ms, spike_recording='population')
        units = {'voltage':u.mV, 'output_voltage':u.mV, 'calcium_mM':u.mM, 'calcium_reversal_mV':u.mV,
                 'ica_inward_ma_cm2':u.mA/u.cm**2, 'ikv3_inward_ma_cm2':u.mA/u.cm**2,
                 'ina_inward_ma_cm2':u.mA/u.cm**2}
        arrays = {'time_ms':np.asarray(result.time.to_decimal(u.ms))+args.dt_ms}
        arrays.update({name:np.asarray(value.to_decimal(units[name]) if name in units else value)
                       for name,value in result.traces['cell_'+identity].items()})
        arrays['events'] = np.asarray(result.spikes['cell_'+identity])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        traces = args.output.with_suffix('.npz')
        np.savez_compressed(traces, **arrays)
        with np.load(baseline_path, allow_pickle=False) as baseline:
            stride = round((.005 if args.dt_ms == .005 else .0025)/args.dt_ms)
            np.testing.assert_allclose(baseline['time_ms'], arrays['time_ms'][stride-1::stride], rtol=0, atol=1e-12)
            old, new = baseline['cell_'+identity+'_voltage'], arrays['voltage'][stride-1::stride]
            finite = np.isfinite(old) & np.isfinite(new)
            delta = float(np.max(np.abs(old[finite]-new[finite])))
            masks_match = bool(np.array_equal(np.isfinite(old), np.isfinite(new)))
        first_bad = {}
        for name, values in arrays.items():
            bad = np.argwhere(~np.isfinite(values))
            if bad.size:
                first_bad[name] = float(arrays['time_ms'][bad[0,0]])
        calcium = arrays['calcium_mM'][:,0]
        nonpositive = np.flatnonzero(np.isfinite(calcium) & (calcium <= 0))
        report = dict(cell_id=identity, scope='experimental calcium integrator; no physiology pass', dt_ms=args.dt_ms, solver='h01_staggered_calcium_implicit',
            exact_cell_metadata_except_solver_match=True, unchanged_geometry=True, n_cv=cell.n_cv,
            output_cv_id=cv_id, output_cv_area_um2=output_area, modeled_soma_area_um2=soma_area,
            input_density_at_output_ma_cm2=float((1.*u.nA/(output_area*u.um**2)).to_decimal(u.mA/u.cm**2)),
            baseline_voltage_max_difference_mv=delta, baseline_finite_masks_match=masks_match,
            all_recorded_arrays_finite=all(np.isfinite(v).all() for v in arrays.values()),
            first_nonfinite_ms=first_bad,
            first_nonpositive_calcium_ms=float(arrays['time_ms'][nonpositive[0]]) if nonpositive.size else None,
            finite_calcium_min_mM=float(calcium[np.isfinite(calcium)].min()),
            peak_rss_mib=process_rss_mb(peak=True), elapsed_seconds=time.perf_counter()-started,
            inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [reference_path, baseline_path, Path(__file__), traces, Path('braintrace/datasets/h01_calcium_solver.py'), Path('braintrace/datasets/h01_calcium_implicit.py')]})
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        emit(json.dumps(report))
        assert report['all_recorded_arrays_finite'] and nonpositive.size == 0, 'Numerical failure persists.'


if __name__ == "__main__":
    main()
