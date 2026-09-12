"""Bounded diagnostic evidence from the co-located synthetic coupling fixture.

Run from the repository root with the scientific development environment:
``python -m docs.biology.neuroglial_probe``. This is not an H01 anatomy factory.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import time

import braincell
import brainstate
import brainunit as u
import numpy as np

from braintrace.biophysics.neuroglial_test import coupled_system, trajectory


def main():
    """Write diagnostic physical responses and code provenance as JSON."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary, brainstate.environ.context(precision=64, dt=.005*u.ms):
        step, coupling, _ = coupled_system(Path(temporary))
        env = coupling.environment
        before = float(np.sum(env.potassium.value*env.k_transport.volumes) +
                       np.sum(env.inside_potassium.value*env.membranes.inside_volumes))
        initial_voltage = float(coupling.astrocytes[0].V.value.to_decimal(u.mV).reshape(()))
        spikes, voltage, calcium = trajectory(step)
        after = float(np.sum(env.potassium.value*env.k_transport.volumes) +
                      np.sum(env.inside_potassium.value*env.membranes.inside_volumes))
        control, _, _ = coupled_system(Path(temporary), gaba_enabled=False)
        _, control_voltage, _ = trajectory(control)
        no_glutamate, _, _ = coupled_system(Path(temporary), glutamate_enabled=False)
        _, _, control_calcium = trajectory(no_glutamate)
        report = dict(
            status='SYNTHETIC_COUPLED_DIAGNOSTIC', basis=coupling.basis,
            neurons=2, astrocytes=1, astrocyte_compartments=1, dt_ms=.005, steps=10,
            duration_ms=.05, neuronal_spike_counts=np.sum(spikes, axis=0).tolist(),
            all_valid=bool(coupling.valid.value), potassium_balance_error_mm_um3=after-before,
            glial_inside_potassium_mm=float(env.inside_potassium.value[-1]),
            glial_initial_voltage_mv=initial_voltage,
            glial_final_voltage_mv=float(coupling.astrocytes[0].V.value.to_decimal(u.mV).reshape(())),
            gaba_released_mm_um3=float(env.gaba_released_amount.value),
            glutamate_released_mm_um3=float(coupling.glutamate.released_amount.value),
            glutamate_removed_mm_um3=float(coupling.glutamate.removed_amount.value),
            glutamate_balance_error_mm_um3=float(coupling.glutamate.balance_error.value),
            gaba_voltage_effect_min_mv=float(np.min(voltage[-1]-control_voltage[-1])),
            glutamate_calcium_effect_max_mm=float(np.max(calcium[-1, :, :4]-control_calcium[-1, :, :4])),
            ip3_final_mm=float(calcium[-1, 0, 24]),
            elapsed_seconds=time.perf_counter()-started,
            limitations=['Synthetic roles, geometry, intracellular neuronal volumes and diagnostic glutamate dose',
                         'Kir-only glial cable; no calcium membrane charge feedback',
                         'No measured H01 glial assembly, long wave or joint learning qualification'])
    root = Path(__file__).resolve().parents[2]
    sources = ['braintrace/biophysics/neuroglial.py', 'braintrace/biophysics/neuroglial_test.py',
               'braintrace/biophysics/cable_kir.py', 'braintrace/biophysics/transmitter.py',
               'docs/biology/neuroglial_probe.py']
    report['implementation_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    output = Path(__file__).parent/'neuroglial-phase/coupled-probe.json'
    output.write_text(json.dumps(report, indent=2)+'\n', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
