"""Diagnostic sodium-availability clamp; does not change production defaults."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import brainstate
import brainunit as u
from braincell.mech import MechanismProbe
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
from braintrace.datasets.h01_pv_channels import _CHANNELS
from braintrace.datasets.h01_ei_profiles import get_ei_profile
from examples.h01_ei_candidates import label_partition

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dt-ms', type=float, default=.005)
parser.add_argument('--region', choices=('all', 'soma', 'axon'), default='all')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert np.isfinite(args.dt_ms) and args.dt_ms > 0
densities = {r[0]: dict(r[3])['NaTg']*1000 for r in get_ei_profile('I').regions if 'NaTg' in dict(r[3])}
assert set(densities) == {'soma', 'axon'} and len(set(densities.values())) == 2
cls = _CHANNELS['NaTg']
original = cls.f_h_inf

def _clamped_equilibrium(self, voltage, *ions):
    equilibrium = original(self, voltage, *ions)
    if not hasattr(self, 'h'):
        return equilibrium
    instant = brainstate.environ.get('t', 0.*u.ms).to_decimal(u.ms)
    active = (instant >= 2.) & (instant < 5.)
    selected = True if args.region == 'all' else jnp.isclose(self.g_max.to_decimal(u.mS/u.cm**2), densities[args.region], rtol=1e-12, atol=0.)
    return jnp.where(active & selected & (equilibrium < self.h.value), self.h.value, equilibrium)

cls.f_h_inf = _clamped_equilibrium
with brainstate.environ.context(precision=64):
    imported = H01Archive(Path('.cache/h01/proofread104.zip')).load('678539249', component=0)
    regions, basis = label_partition(imported)
    cell, report = make_h01_ei_cell(imported, H01Annotations(Path('.cache/h01')),
        polarity='I', regions=regions, region_basis=basis, current_na=1.,
        delay_ms=2., duration_ms=3., max_cv_length_um=10.)
    soma = imported.anatomy().soma_location()
    for gate in ('m', 'h'):
        cell.place(soma, MechanismProbe(mechanism='pv_NaTg', field=gate, name=gate))
    print('Inactivation split constructed', flush=True)
    result = cell.run(dt=args.dt_ms*u.ms, duration=10.*u.ms)
    voltage = np.asarray(result.traces['voltage'].to_decimal(u.mV)).ravel()
    m, h = (np.asarray(result.traces[g]).ravel() for g in ('m', 'h'))
    time = args.dt_ms*np.arange(1, len(voltage)+1)
    assert all(np.isfinite(x).all() for x in (voltage, m, h))
    window = (time > 2.) & (time <= 5.)
    if args.region in ('all', 'soma'):
        assert np.all(np.diff(h[window]) >= -1e-12)
    report.update(dt_ms=args.dt_ms, current_na=1., intervention='NaTg h closing blocked in '+args.region+' during 2 <= t < 5 ms',
        selected_region=args.region, selector_densities_ms_cm2=densities,
        peak_mv=float(voltage.max()), peak_ms=float(time[np.argmax(voltage)]),
        qualification='Nonphysical diagnostic intervention; not a promoted candidate or human spike qualification.')
    if args.dt_ms == .005:
        baseline = np.load('docs/evidence/h01-i-local-currents.npz')['voltage_mv']
        unchanged = np.array_equal(voltage[time <= 2.], baseline[time <= 2.])
        assert unchanged
        report['preintervention_voltage_exact'] = unchanged
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), time_ms=time, voltage_mv=voltage, NaTg_m=m, NaTg_h=h)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print({'peak_mv': report['peak_mv'], 'peak_ms': report['peak_ms']}, flush=True)
