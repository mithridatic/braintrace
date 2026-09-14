"""Fit the frozen within-sweep availability candidate to all original current samples."""

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[4]))
from docs.evidence.h01_l1_recovery_state import fit_state_currents

if (root/'joint-fit.json').exists() or (root/'joint-fit.npz').exists():
    raise FileExistsError('Joint-fit outputs already exist in this prefix.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/'inputs.json').read_bytes())
assert sha(root/'inputs.npz')==m['arrays_sha256']
with np.load(root/'inputs.npz') as a:a=dict(a)
started=time.perf_counter()
arrays,metadata=fit_state_currents(a['gaps_ms'],a['pulse_phase_ms'],a['post_phase_ms'],
                                  a['first_current_pa'],a['second_current_pa'],a['post_current_pa'])
metadata['elapsed_seconds']=time.perf_counter()-started
np.savez_compressed(root/'joint-fit.npz',**arrays)
metadata.update(arrays_sha256=sha(root/'joint-fit.npz'),
                 inputs=dict(path='inputs.json',sha256=sha(root/'inputs.json')),
                 source_sha256=m['source_sha256'])
(root/'joint-fit.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:metadata[k] for k in ('parameters','optimizer_success','evaluations','elapsed_seconds',
                                        'active_bounds','scaled_jacobian_singular_values','scaled_jacobian_rank')},indent=2))
print({k:v['all_rms_pa'] for k,v in metadata['residuals'].items()})
