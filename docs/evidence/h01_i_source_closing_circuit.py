"""Run the existing H01 diagnostic circuit with one explicit I gate override."""
import json
import runpy
import sys
from pathlib import Path
import jax.numpy as jnp
from braintrace.datasets.h01_pv_channels import _CHANNELS

cls=_CHANNELS['NaTg']
_original=cls.f_h_tau

def _restored(self,voltage,*ions):
    value=_original(self,voltage,*ions)
    equilibrium=self._rates(voltage,ions)['h'][0]
    if not hasattr(self,'h'):
        return value
    return jnp.where(equilibrium<=self.h.value,value/self.phase_factors['h'][1],value)

cls.f_h_tau=_restored
runpy.run_module('examples.h01_ei_circuit',run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.json')
record=json.loads(output.read_text())
record['cells']['I']['diagnostic_override']={
    'NaTg_h_closing_factor':1.,'candidate_value':.15,
    'scope':'all I NaTg regions; all times; E unchanged'}
record['qualification']='Functional diagnostic with source closing-time restoration; not human qualified or promoted.'
output.write_text(json.dumps(record,indent=2)+'\n')
