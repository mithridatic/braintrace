"""Show all individual human recovery measurements and matched model cycles."""

from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parent
r = json.loads((out/'result.json').read_text())
d = np.load(out/'model-observations.npz')
names = sorted({x['filename'] for x in r['records']})
volts = d['voltage_mv']
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
errors = np.full((19, 4), np.nan)
for i, name in enumerate(names):
    points = sorted([x for x in r['records'] if x['filename'] == name and x['eligible']], key=lambda x: x['voltage_mv'])
    axes[0].plot([p['voltage_mv'] for p in points], [p['observed_tau_ms'] for p in points],
                 '.-', lw=.7, alpha=.75, label=name)
    for point in points:
        errors[i, np.flatnonzero(volts == point['voltage_mv'])[0]] = point['signed_error_ms']
axes[0].plot(volts, [p['current_recovery_fit']['tau_ms'] for p in r['predictions']], 'k-o', lw=2, label='Fixed model, fitted current recovery')
axes[0].set_xlabel('Recovery command voltage (mV)'); axes[0].set_ylabel('Fitted recovery time constant (ms)')
axes[0].set_title('All nineteen human recordings; no averaging'); axes[0].grid(alpha=.2)
limit = abs(errors).max()
im = axes[1].imshow(errors, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
axes[1].set_xticks(range(4), labels=[str(int(v)) for v in volts])
axes[1].set_yticks(range(19), labels=[n.removesuffix('.nwb') for n in names], fontsize=8)
axes[1].set_xlabel('Recovery command voltage (mV)'); axes[1].set_title('Model minus each human observation')
fig.colorbar(im, ax=axes[1], label='Signed error (ms)')
fig.tight_layout(); fig.savefig(out/'all-human-recovery.png', dpi=130)

fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True, sharey='col')
for row, v in enumerate(volts):
    for delay_index in [0, 1, 11]:
        label = f"Recovery {d['delays_ms'][delay_index]:g} ms"
        axes[row, 0].plot(d['second_pulse_time_ms'], d['inward_proxy'][row, delay_index], label=label)
        axes[row, 1].plot(d['second_pulse_time_ms'], d['second_pulse_states'][row, delay_index, :, 0], label=label)
        axes[row, 2].plot(d['second_pulse_time_ms'], d['second_pulse_states'][row, delay_index, :, 1], label=label)
    for ax, label in zip(axes[row], ['m^3 h (141-V), mV; current proxy', 'Activation fraction m', 'Availability fraction h']):
        ax.set_title(f'{v:g} mV recovery'); ax.set_ylabel(label)
        ax.set_xlabel('Time from second-pulse onset (ms)'); ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.suptitle('Fixed 25 C model: second-pulse states; human raw currents unavailable')
fig.tight_layout(); fig.savefig(out/'model-recovery-cycles.png', dpi=130)
