"""Record direct CV currents with a compiled BrainCell loop."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import brainstate
import brainunit as u
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_annotations import H01Annotations
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
from braintrace.datasets.h01_ei_profiles import get_ei_profile
from braintrace.datasets.h01_pv_channels import _CHANNELS
from examples.h01_ei_candidates import label_partition
from braincell._multi_compartment import bridge, currents
from braincell._multi_compartment.probes import _representative_cv_id
from braincell.mech import StateProbe
from braincell.quad._staggered import _linear_and_const_term

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dt-ms',type=float,default=.005)
parser.add_argument('--region',choices=('none','soma','axon'),default='none')
parser.add_argument('--output',type=Path,default=Path('docs/evidence/h01-i-current-balance'))
args=parser.parse_args()
dt=args.dt_ms
assert dt in (.005,.000625)
if args.region != 'none':
    densities={r[0]:dict(r[3])['NaTg']*1000 for r in get_ei_profile('I').regions if 'NaTg' in dict(r[3])}
    assert set(densities)=={'soma','axon'} and len(set(densities.values()))==2
    cls=_CHANNELS['NaTg'];original=cls.f_h_inf
    def _clamped(self,voltage,*ions):
        equilibrium=original(self,voltage,*ions)
        if not hasattr(self,'h'): return equilibrium
        t=brainstate.environ.get('t',0.*u.ms).to_decimal(u.ms)
        selected=jnp.isclose(self.g_max.to_decimal(u.mS/u.cm**2),densities[args.region],rtol=1e-12,atol=0.)
        return jnp.where((t>=2.)&(t<5.)&selected&(equilibrium<self.h.value),self.h.value,equilibrium)
    cls.f_h_inf=_clamped
with brainstate.environ.context(precision=64, dt=dt*u.ms):
    imported=H01Archive(Path('.cache/h01/proofread104.zip')).load('678539249',component=0)
    regions,basis=label_partition(imported)
    cell,record=make_h01_ei_cell(imported,H01Annotations(Path('.cache/h01')),polarity='I',regions=regions,region_basis=basis,current_na=1.)
    cell.init_state()
    runtime=cell.runtime
    layouts=[l for l in runtime.layouts if isinstance(runtime.get_layout_mechanism(l.id),StateProbe)]
    assert len(layouts)==1
    point=int(layouts[0].point_index[0]);cv_id=_representative_cv_id(runtime,point_id=point)
    assert int(runtime.node_tree.cv_to_mid_node_id[cv_id])==point
    area=cell.cvs[cv_id].area
    capacitance=float((cell.C[cv_id]*area).to_decimal(u.nF))
    row=np.asarray(runtime.axial_operator_np[cv_id]);neighbors=np.flatnonzero((np.arange(len(row))!=cv_id)&(np.abs(row)>1e-12))
    conductance=-row[neighbors]*capacitance
    assert np.all(conductance>0) and abs(row.sum())<1e-7
    def _step(i):
        with brainstate.environ.context(t=i*dt*u.ms):
            old_voltage=cell.V.value
            linear,constant=_linear_and_const_term(cell,old_voltage)
            cell.update()
        instant=(i+1)*dt*u.ms
        v=cell.V.value.to_decimal(u.mV)
        total=(currents.total_membrane_current(cell,V_cv=cell.V.value,t=instant)[cv_id]*area).to_decimal(u.nA)
        applied=(bridge.point_to_cv(currents._clamp_density(runtime,t=instant),runtime)[cv_id]*area).to_decimal(u.nA)
        axial=conductance*(v[neighbors]-v[cv_id])
        discrete_membrane=((linear[cv_id]*cell.V.value[cv_id]+constant[cv_id])*cell.C[cv_id]*area).to_decimal(u.nA)
        discrete_capacitive=capacitance*(v[cv_id]-old_voltage.to_decimal(u.mV)[cv_id])/dt
        return v[cv_id],v[neighbors],total,applied,axial,discrete_membrane,discrete_capacitive
    print('Current balance initialized',flush=True)
    result=brainstate.transform.for_loop(_step,jnp.arange(round(10./dt)))
    voltage,neighbor_v,total,applied,axial,discrete_membrane,discrete_capacitive=map(np.asarray,result)
    time=dt*np.arange(1,len(voltage)+1)
    baseline=np.load('docs/evidence/h01-i-local-currents.npz')['voltage_mv'] if dt==.005 else np.load('docs/evidence/h01-i-defaults-spike-probe-fine.npz')['I_voltage_mv']
    if args.region!='none':
        assert dt==.005
        baseline=np.load('docs/evidence/h01-i-inactivation-'+args.region+'.npz')['voltage_mv']
    error=float(np.max(abs(voltage-baseline)))
    assert error==0.
    capacitive=capacitance*np.gradient(voltage,dt)
    residual=capacitive-total-axial.sum(axis=1)
    discrete_residual=discrete_capacitive-discrete_membrane-axial.sum(axis=1)
    valid=(time>.01)&(time<9.99)&(abs(time-2.)>.01)&(abs(time-5.)>.01)
    assert all(np.isfinite(a).all() for a in result)
    record.update(intervention_region=args.region,cv_id=cv_id,point_id=point,point_is_cv_midpoint=True,area_um2=float(area.to_decimal(u.um**2)),capacitance_nf=capacitance,neighbor_cv_ids=neighbors.tolist(),coupling_us=conductance.tolist(),baseline_error_mv=error,residual_max_na=float(np.max(abs(residual[valid]))),residual_limit_na=1e-5,discrete_residual_max_na=float(np.max(abs(discrete_residual))),dt_ms=dt,current_sign='inward positive',qualification='End-step current observations with central voltage difference; numerical diagnostic, not biological validation.')
    p=args.output
    np.savez_compressed(p.with_suffix('.npz'),time_ms=time,voltage_mv=voltage,neighbor_voltage_mv=neighbor_v,total_membrane_na=total,applied_na=applied,axial_inward_na=axial,capacitive_na=capacitive,residual_na=residual,discrete_membrane_na=discrete_membrane,discrete_capacitive_na=discrete_capacitive,discrete_residual_na=discrete_residual)
    p.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print({k:record[k] for k in ['cv_id','point_is_cv_midpoint','baseline_error_mv','residual_max_na','discrete_residual_max_na']},flush=True)
