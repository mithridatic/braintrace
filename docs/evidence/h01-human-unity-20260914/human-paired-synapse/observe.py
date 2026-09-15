"""Retain and render the prespecified human pair without fitting a model."""

from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyabf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / 'docs' / 'evidence'))
from h01_paired_response import electrode_index, spike_windows

receipt = json.loads((OUT / 'acquisition.json').read_text())
source = next(f for f in receipt['files'] if f['id'] == 359652)
path = ROOT / source['path']
assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
with (OUT / 'Data_fig2_Pyr_FS.csv').open(newline='', encoding='utf-8-sig') as handle:
    rows = list(csv.DictReader(handle))
pair, = [r for r in rows if r['File/folder name'] == path.name]
assert pair['species'] == 'Human' and pair['Type'] == 'EXC'
abf = pyabf.ABF(str(path))
pre_index = electrode_index(abf.adcNames, abf.adcUnits, pair['SpikeCh'])
post_index = electrode_index(abf.adcNames, abf.adcUnits, pair['ResponseCh'])
sweeps = list(dict.fromkeys([0, 1, abf.sweepCount - 1]))
arrays = {}
for sweep in sweeps:
    abf.setSweep(sweep, channel=pre_index)
    arrays[f's{sweep}_t'] = abf.sweepX.copy()
    arrays[f's{sweep}_pre'] = abf.sweepY.copy()
    arrays[f's{sweep}_parser_command_pre'] = abf.sweepC.copy()
    abf.setSweep(sweep, channel=post_index)
    assert np.array_equal(arrays[f's{sweep}_t'], abf.sweepX)
    arrays[f's{sweep}_post'] = abf.sweepY.copy()
    arrays[f's{sweep}_parser_command_post'] = abf.sweepC.copy()
    for side in ['pre', 'post']:
        assert np.isfinite(arrays[f's{sweep}_{side}']).all()

if sys.argv[1:] == ['full']:
    np.savez_compressed(OUT / 'full-sweeps.npz', **arrays)
    meta = dict(source=source, pair=pair, pyabf_version=pyabf.__version__,
                adc_names=abf.adcNames, adc_units=abf.adcUnits,
                dac_names=abf.dacNames, dac_units=abf.dacUnits,
                sample_rate_hz=abf.dataRate, sweep_count=abf.sweepCount,
                samples_per_sweep=abf.sweepPointCount, selected_sweeps=sweeps,
                protocol=abf.protocol, protocol_path=abf.protocolPath,
                recorded_datetime=abf.abfDateTimeString,
                command_mapping='Parser-associated reconstructed commands; physical electrode mapping not verified.',
                scores_promoted=False, physiology_qualified=False)
    (OUT / 'recording.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(len(sweeps), 2, figsize=(12, 8), sharex=True, sharey='col')
    for row, sweep in enumerate(sweeps):
        for col, side in enumerate(['pre', 'post']):
            axes[row, col].plot(arrays[f's{sweep}_t'], arrays[f's{sweep}_{side}'], lw=.5)
            axes[row, col].set_title(f'Sweep {sweep}: {side} {pair["SpikeCh" if side == "pre" else "ResponseCh"]}')
            axes[row, col].set_ylabel('Recorded voltage (mV)')
            axes[row, col].grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Within-sweep time (s)')
    fig.suptitle('Human paired recording 2015_11_04_0076: all original samples')
    fig.tight_layout()
    fig.savefig(OUT / 'full-sweeps.png', dpi=140)
    print(json.dumps(meta, indent=2))
elif sys.argv[1:] == ['events']:
    assert (OUT / 'full-sweeps.png').is_file(), 'Inspect full sweeps first.'
    events, descriptions, shown = {}, [], []
    for sweep in sweeps:
        result = spike_windows(*(arrays[f's{sweep}_{k}'] for k in ['t', 'pre', 'post']),
                               rate_hz=abf.dataRate)
        events.update({f's{sweep}_{k}': v for k, v in result.items()})
        count = len(result['accepted_crossings'])
        selected = list(dict.fromkeys([0, 1, count-1])) if count >= 2 else list(range(count))
        descriptions.append(dict(sweep=sweep, crossings=result['crossing_indices'].tolist(),
                                 omitted_boundary_crossings=result['boundary_crossings'].tolist(),
                                 plotted_event_ordinals=selected))
        shown.extend((sweep, i, result) for i in selected)
    np.savez_compressed(OUT / 'events.npz', **events)
    (OUT / 'events.json').write_text(json.dumps(descriptions, indent=2)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(len(sweeps), 3, figsize=(15, 9), squeeze=False)
    pre_limits = [min(a[f's{s}_pre'].min() for s in sweeps) for a in [arrays]]
    pre_limits += [max(arrays[f's{s}_pre'].max() for s in sweeps)]
    post_limits = [min(arrays[f's{s}_post'].min() for s in sweeps),
                   max(arrays[f's{s}_post'].max() for s in sweeps)]
    for ax in axes.flat:
        ax.set_visible(False)
    for sweep, ordinal, result in shown:
        row = sweeps.index(sweep)
        col = descriptions[row]['plotted_event_ordinals'].index(ordinal)
        ax = axes[row, col]
        ax.set_visible(True)
        crossing = result['accepted_crossings'][ordinal]
        t = (result['time_s'][ordinal] - arrays[f's{sweep}_t'][crossing]) * 1000
        ax.plot(t, result['pre_mv'][ordinal], color='tab:blue', lw=.8)
        ax.set_ylim(pre_limits[0]-2, pre_limits[1]+2)
        ax.set_ylabel('Presynaptic mV', color='tab:blue')
        twin = ax.twinx()
        twin.plot(t, result['post_mv'][ordinal], color='tab:orange', lw=.8)
        twin.set_ylim(post_limits[0]-.1, post_limits[1]+.1)
        twin.set_ylabel('Postsynaptic mV', color='tab:orange')
        ax.set_title(f'Sweep {sweep}, event {ordinal}, source index {crossing}')
        ax.set_xlabel('Time from presynaptic upward 0 mV crossing (ms)')
        ax.axvline(0, color='gray', lw=.5)
        ax.grid(alpha=.2)
    fig.suptitle('Original paired voltage samples; separate fixed scales for pre and post')
    fig.tight_layout()
    fig.savefig(OUT / 'paired-events.png', dpi=140)
    print(json.dumps(descriptions, indent=2))
else:
    raise SystemExit('Specify full or events.')
