"""Export all matched human conditioning tests without opening external currents."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[4]))
from docs.evidence.h01_l1_voltage_clamp import read_sweep
from docs.evidence.h01_l1_conditioning import paired_conditioning, match_commands

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--nwb', type=Path, required=True)
args = parser.parse_args()
if list(ROOT.glob('sweep-*.json')) or list(ROOT.glob('pair-*.json')):
    raise FileExistsError('This evidence prefix already contains source exports.')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
arrays, metadata, commands = {}, {}, {}
for sweep in [*range(70,79), *range(89,98)]:
    a, m = read_sweep(args.nwb, sweep, session='840043481')
    prefix = ROOT / f'sweep-{sweep}'
    np.savez_compressed(prefix.with_suffix('.npz'), **a)
    m['arrays_sha256'] = digest(prefix.with_suffix('.npz'))
    prefix.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8',newline='\n')
    t = a['time_ms']
    levels = np.unique(a['command_voltage_mv'][(t >= 1100) & (t < 2100)])
    if len(levels) != 1:
        raise ValueError(f'Sweep {sweep} has a nonconstant test command.')
    arrays[sweep], metadata[sweep], commands[sweep] = a, m, float(levels[0])
pairs = match_commands({s:commands[s] for s in range(70,79)},
                       {s:commands[s] for s in range(89,98)})
inventory = []
for total, conditioned in pairs:
    a, m = paired_conditioning(arrays[total],arrays[conditioned])
    prefix = ROOT / f'pair-{total}-{conditioned}'
    np.savez_compressed(prefix.with_suffix('.npz'), **a)
    m.update(total_sweep=total,conditioned_sweep=conditioned,
             source_exports=[dict(path=f'sweep-{s}.json',sha256=digest(ROOT/f'sweep-{s}.json'))
                             for s in (total,conditioned)],
             arrays_sha256=digest(prefix.with_suffix('.npz')))
    prefix.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8',newline='\n')
    inventory.append(dict(total=total,conditioned=conditioned,test_command_mv=m['test_command_mv'],
                          baseline_difference_pa=m['baseline_difference_pa']))
(ROOT/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(inventory,indent=2))
