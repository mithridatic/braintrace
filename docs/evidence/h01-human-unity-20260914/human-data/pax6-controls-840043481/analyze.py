"""Compare stored human controls and transfer a control-only fitted gain."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[4]))
from docs.evidence.h01_l1_control import extract_test_pulse,gain_transfer

if (root/'inventory.json').exists():
    raise FileExistsError('This control-analysis prefix already contains results.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin='30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944'
full,controls,control_meta,inventory={},{},{},[]
for sweep in range(70,98):
    folder=root.parent/('pax6-conditioning-840043481' if sweep<79 or sweep>=89 else 'pax6-840043481')
    receipt=json.loads((folder/'analysis-receipt.json').read_bytes())
    path=folder/f'sweep-{sweep}.npz';jpath=path.with_suffix('.json')
    assert sha(path)==receipt['files'][path.name]
    assert sha(jpath)==receipt['files'][jpath.name]
    m=json.loads(jpath.read_bytes());assert m['source_sha256']==pin
    with np.load(path) as a:full[sweep]=dict(a)
    a,c=extract_test_pulse(full[sweep]);controls[sweep]=a;control_meta[sweep]=c
    prefix=root/f'control-{sweep}'
    np.savez_compressed(prefix.with_suffix('.npz'),**a)
    c.update(sweep=sweep,family=m['family'],absolute_start_s=m['absolute_start_s'],
             instrument=m['instrument'],source=dict(path='../'+folder.name+'/'+jpath.name,sha256=sha(jpath)),
             source_arrays=dict(path='../'+folder.name+'/'+path.name,sha256=sha(path)),
             arrays_sha256=sha(prefix.with_suffix('.npz')))
    prefix.with_suffix('.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8',newline='\n')
    inventory.append(c)

def main_response(sweep):
    """Extract the independently observed large command, retaining its source clock.

    Parameters
    ----------
    sweep : int
        Total sweep 72 or a conditioning sweep 89-97.

    Returns
    -------
    dict
        Original and baseline-centered first 100 ms of the actual -20 mV command.
    """
    a=full[sweep];t=a['time_ms'];i=a['total_current_pa'];v=a['command_voltage_mv']
    onset=1100. if sweep==72 else 1000.
    w=(t>=onset)&(t<onset+100);b=(t>=900)&(t<990)
    baseline=float(i[b].mean())
    assert np.allclose(v[w],-19.987133098766208,atol=1e-7,rtol=0)
    return dict(phase_ms=t[w]-onset,source_time_ms=t[w],command_voltage_mv=v[w],
                centered_current_pa=i[w]-baseline,total_current_pa=i[w],
                baseline_time_ms=t[b],baseline_current_pa=i[b])

reference=main_response(72)
results=[]
for sweep in range(89,98):
    target=main_response(sweep)
    a,m=gain_transfer(controls[72],controls[sweep],reference,target)
    for prefix,response in [('reference',reference),('target',target)]:
        for key,value in response.items():a[f'{prefix}_main_{key}']=value
    points=[]
    for phase in (20.,60.,90.):
        k=int(np.flatnonzero(np.isclose(a['main_phase_ms'],phase,rtol=0,atol=1e-8))[0])
        points.append(dict(phase_ms=phase,reference_pa=float(a['main_reference_pa'][k]),
                           target_pa=float(a['main_target_pa'][k]),predicted_pa=float(a['main_predicted_pa'][k]),
                           residual_pa=float(a['main_residual_pa'][k])))
    p=root/f'gain-72-{sweep}'
    np.savez_compressed(p.with_suffix('.npz'),**a)
    m.update(reference_sweep=72,target_sweep=sweep,points=points,
             source_controls=[dict(path=f'control-{s}.json',sha256=sha(root/f'control-{s}.json')) for s in (72,sweep)],
             arrays_sha256=sha(p.with_suffix('.npz')))
    p.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8',newline='\n')
    results.append(m)
(root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8',newline='\n')
(root/'gain-summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8',newline='\n')
for row in results:
    print(row['target_sweep'],'gain',round(row['gain'],4),'control RMS',round(row['control_fit_rms_pa'],3),
          'main post5 RMS',round(row['main_post5_residual_rms_pa'],3),'main residual at20',round(row['points'][0]['residual_pa'],3))
