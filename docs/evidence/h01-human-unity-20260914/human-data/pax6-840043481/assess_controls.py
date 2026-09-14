"""Retain conditional subtraction and direct PAX6 control observations."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from docs.evidence.h01_l1_leak import linear_step_estimate


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'preparation.json').read_bytes())
plt.rcParams.update({'path.simplify': False, 'font.size': 9})


def load(sweep):
    """Load a hash-verified same-session export.

    Parameters
    ----------
    sweep : int
        Registered PAX6 calibration sweep.

    Returns
    -------
    dict
        Original sample arrays.
    """
    path = ROOT / f'sweep-{sweep}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST['files'][path.name]
    meta = json.loads((ROOT / f'sweep-{sweep}.json').read_bytes())
    assert meta['session_id'] == '840043481' and meta['source_sha256'] == MANIFEST['nwb_sha256']
    with np.load(path) as a:
        return dict(a)


families = [('Total', [70, 74, 78]), ('Prepulse', [89, 93, 97]),
            ('Sustained', [104, 108, 112])]
fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)
limits = []
for row, (family, sweeps) in enumerate(families):
    for sn, color in zip(sweeps, ['#176b9b', '#b16c00', '#9c2b51']):
        a = load(sn)
        t, current, command = a['time_ms'], a['total_current_pa'], a['command_voltage_mv']
        w = (t >= 990) & (t <= 2200)
        early = (t >= 1099) & (t <= 1120)
        axes[row, 0].plot(t[w], current[w], lw=.5, color=color, label=f'Sweep {sn}')
        axes[row, 1].plot(t[w], command[w], lw=1, color=color)
        axes[row, 2].plot(t[early], current[early], lw=.5, color=color)
        limits.extend([float(current[w].min()), float(current[w].max())])
    axes[row, 0].set_title(family)
    axes[row, 0].legend()
    axes[row, 1].set_ylim(-105, 85)
for row in range(3):
    for col in [0, 2]:
        axes[row, col].set_ylim(min(limits) * 1.1, max(limits) * 1.1)
        axes[row, col].set_ylabel('Total amplifier current (pA)')
    axes[row, 1].set_ylabel('DAC + holding command (mV)')
for ax in axes.flat:
    ax.set_xlabel('Original sweep time (ms)')
    ax.grid(alpha=.2)
fig.suptitle('Human PAX6 CDH12 specimen 840043506 | unfiltered current | no junction or leak correction')
fig.savefig(ROOT / 'raw-steps.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
records = []
for row, (label, controls, target) in enumerate([
        ('Holding near -90 mV', range(79, 89), 78),
        ('Holding near -20 mV', range(113, 123), 112)]):
    control_records = []
    for idx, sn in enumerate(controls):
        a = load(sn)
        t, current = a['time_ms'], a['total_current_pa']
        baseline = (t >= 900) & (t < 990)
        offset = float(current[baseline].mean())
        centered = current - offset
        color = plt.get_cmap('viridis')(idx / 9)
        for col, window in enumerate([(1099, 2100), (1099, 1110), (2000, 2090)]):
            w = (t >= window[0]) & (t < window[1])
            axes[row, col].plot(t[w], centered[w], lw=.45, color=color,
                                label=str(sn) if col == 0 else None)
        late = (t >= 2000) & (t < 2090)
        control_records.append(dict(sweep=sn, baseline_mean_pa=offset,
                                    late_centered_mean_pa=float(centered[late].mean())))
    for sn in [controls.start, controls.stop - 1]:
        arrays, metadata = linear_step_estimate(load(target), load(sn))
        prefix = ROOT / f'conditional-{target}-control-{sn}'
        if prefix.with_suffix('.npz').exists() or prefix.with_suffix('.json').exists():
            raise FileExistsError('Use a new analysis prefix; retained estimates cannot be overwritten.')
        with prefix.with_suffix('.npz').open('xb') as f:
            np.savez_compressed(f, **arrays)
        metadata.update(signal_sweep=target, control_sweep=sn,
                        nwb_sha256=MANIFEST['nwb_sha256'],
                        arrays_sha256=hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest())
        prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        axes[row, 3].plot(arrays['time_ms'], arrays['conditional_current_pa'], lw=.5,
                          label=f'Control {sn}')
    axes[row, 0].set_title(label)
    axes[row, 0].legend(ncol=2, fontsize=7)
    axes[row, 3].legend()
    records.append(dict(holding_label=label, target_sweep=target, controls=control_records))
for col in range(3):
    limits = [ax.get_ylim() for ax in axes[:, col]]
    for ax in axes[:, col]: ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
for row in range(2):
    for col in range(3): axes[row, col].set_ylabel('Baseline-centered control current (pA)')
    axes[row, 3].set_ylabel('Conditional difference (pA)')
for ax in axes.flat:
    ax.set_xlabel('Original sweep time (ms)')
    ax.grid(alpha=.2)
axes[0, 1].set_title('Control onset; every sample')
axes[0, 2].set_title('Late control; every sample')
axes[0, 3].set_title('Same target, first versus last control')
fig.suptitle('PAX6 repeated -10 mV controls | conditional linear subtraction does not establish ionic isolation')
fig.savefig(ROOT / 'controls.png', dpi=150)
plt.close(fig)
(ROOT / 'control-observations.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
