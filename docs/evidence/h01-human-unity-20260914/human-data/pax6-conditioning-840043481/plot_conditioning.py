"""Show all matched conditioning currents, commands and differences directly."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
plt.rcParams.update({'path.simplify':False,'font.size':9})
inventory=json.loads((root/'inventory.json').read_bytes())
fig, axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
diff_fig, diff_axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
cmd_fig, cmd_axes=plt.subplots(3,3,figsize=(15,9),constrained_layout=True)
for row, ax, dax, cax in zip(inventory,axes.flat,diff_axes.flat,cmd_axes.flat):
    prefix=root/f'pair-{row["total"]}-{row["conditioned"]}'
    m=json.loads(prefix.with_suffix('.json').read_bytes())
    assert hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest()==m['arrays_sha256']
    with np.load(prefix.with_suffix('.npz')) as a:
        phase=a['phase_ms']
        ax.plot(phase,a['total_current_pa'],lw=.5,label='Total protocol')
        ax.plot(phase,a['conditioned_current_pa'],lw=.5,label='After conditioning')
        mask=(phase>=10)&(phase<980)
        dax.plot(phase[mask],a['difference_pa'][mask],lw=.5,label='Total minus conditioned')
        dax.plot(phase[mask],a['difference_pa'][mask]-m['baseline_difference_pa'],lw=.5,label='Baseline sensitivity')
    for sweep,label in [(row['total'],'Total protocol'),(row['conditioned'],'Conditioned')]:
        with np.load(root/f'sweep-{sweep}.npz') as a:
            mask=(a['time_ms']>=900)&(a['time_ms']<2200)
            cax.plot(a['time_ms'][mask],a['command_voltage_mv'][mask],lw=.9,label=label)
    for axis in (ax,dax,cax):
        axis.set_title(f'{row["test_command_mv"]:.3f} mV | sweeps {row["total"]}, {row["conditioned"]}')
        axis.grid(alpha=.2)
    ax.set_ylim(-100,1800); dax.set_ylim(-100,260); cax.set_ylim(-100,80)
    ax.set_ylabel('Total amplifier current (pA)'); dax.set_ylabel('Difference (pA)')
    ax.set_xlabel('Original test phase (ms)'); dax.set_xlabel('Original test phase (ms)')
    cax.set_xlabel('Original sweep time (ms)'); cax.set_ylabel('Amplifier command (mV)')
axes[0,0].legend(); diff_axes[0,0].legend(fontsize=8); cmd_axes[0,0].legend()
fig.suptitle('All human PAX6 test-voltage pairs | onset transients retained | no ionic isolation claim')
diff_fig.suptitle('Conditioning differences beyond phase 10 ms | original samples | baseline sensitivity is conditional')
cmd_fig.suptitle('Match the 1100-2100 ms test window | conditioning occupies 1000-1100 ms')
for figure,name in [(fig,'paired-currents.png'),(diff_fig,'differences.png'),(cmd_fig,'commands.png')]:
    figure.savefig(root/name,dpi=130)
plt.close('all')
