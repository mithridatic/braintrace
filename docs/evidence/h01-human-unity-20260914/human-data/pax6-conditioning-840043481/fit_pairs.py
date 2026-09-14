"""Retain conditional decay fits and their failures for every matched command."""

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[4]))
from docs.evidence.h01_l1_conditioning import fit_difference

if list(root.glob('fit-*.json')):
    raise FileExistsError('Fit outputs already exist in this evidence prefix.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inventory=json.loads((root/'inventory.json').read_bytes())
results=[]
for row in inventory:
    prefix=root/f'pair-{row["total"]}-{row["conditioned"]}'
    m=json.loads(prefix.with_suffix('.json').read_bytes())
    assert sha(prefix.with_suffix('.npz'))==m['arrays_sha256']
    with np.load(prefix.with_suffix('.npz')) as a:
        phase=a['phase_ms']; original=a['difference_pa']
    for scenario,shift in [('unadjusted',0.),('baseline-sensitivity',m['baseline_difference_pa'])]:
        start=time.perf_counter()
        a,f=fit_difference(phase,original-shift)
        f.update(elapsed_seconds=time.perf_counter()-start,scenario=scenario,
                 subtracted_baseline_difference_pa=shift,test_command_mv=m['test_command_mv'],
                 total_sweep=row['total'],conditioned_sweep=row['conditioned'],
                 input_pair=dict(path=prefix.with_suffix('.json').name,sha256=sha(prefix.with_suffix('.json'))))
        output=root/f'fit-{row["total"]}-{row["conditioned"]}-{scenario}'
        np.savez_compressed(output.with_suffix('.npz'),**a)
        f['arrays_sha256']=sha(output.with_suffix('.npz'))
        output.with_suffix('.json').write_text(json.dumps(f,indent=2)+'\n',encoding='utf-8',newline='\n')
        results.append(f)
(root/'fit-summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8',newline='\n')
for r in results:
    print(r['total_sweep'],r['scenario'],np.round(r['parameters'],3),
          'RMS',round(r['residual_rms_pa'],3),'bounds',r['active_bounds'])
