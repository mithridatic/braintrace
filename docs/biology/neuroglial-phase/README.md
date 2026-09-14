# Neuron-glia coupling phase

Implemented on `feat/h01-braincell-biology`, after spine commit `647d24b`.
This phase provides an executable `NeuroGlialCoupling` owner for `H01NetworkStep`.
It does not add the complete glial spatial manifest to `H01Session.build`.

Actual inhibitory/excitatory population spikes feed separately declared GABA
and glutamate sites. The glial cable has dynamic voltage and the pinned source
Kir4.1 law, with shared K pools, conservative membrane exchange and dynamic
reversal. Transmitter diffusion/uptake feeds mapped astrocyte IP3/calcium; GABA
feeds the neuronal tonic receptor on the next cable tick. Glia remain outside
the neuronal ARC event domain. Biology and geometry remain fixed.

The driver samples all K currents before neuronal/glial advancement, advances
glial voltage after the neurons, applies each saved K exchange once, advances
transmitter fields, then solves calcium. Calcium membrane charge feedback is
not included. Glutamate dose and clearance are required explicit modeling
choices; a source bath concentration is not treated as a measured vesicle dose.

## Evidence and reproduction

`coupled-probe.json` records a 0.05 ms synthetic two-neuron/one-glial-CV diagnostic
at 0.005 ms steps. It reports actual spikes, K conservation and uptake, glial
voltage, transmitter ledgers and matched no-GABA/no-glutamate effects. The probe
deliberately imports the regression fixture; it is not a measured H01 assembly.
Run `python -m docs.biology.neuroglial_probe` in the scientific development env.

`kir-reference.json` records a separate 0.1 ms fixed-pool NEURON 8.2.7 cable
using the pinned original `Kir4.mod`. Source and freshly built library hashes
are recorded. The test compares all 20 evolved BrainCell voltage samples within
0.001 mV. The source's shifted voltage law is retained; its zero-current point
is not simply the Nernst reversal. Reproduce in a NEURON environment with:

```
python docs/biology/neuron_kir_reference.py --build --mechanisms /var/tmp/h01-kir-reference --output docs/biology/neuroglial-phase/kir-reference.json
```

`verification.xml`, `coverage.json` and `verification-summary.json` record the
affected CPU gate and new production-module coverage. Tests include a compiled
continuous K-to-glial-voltage derivative against central finite differences,
exact reset, and save/rebuild/restore followed by exact replay of every packed
physical state, including glial voltage, calcium and both release RNG streams.
This derivative test establishes a continuous physical path, not a gradient
through hard spike decisions or a joint pp-prop learning-rule qualification.

Recorded result: **185 passed, six existing session readout-state warnings**;
all three new production modules have **100% statement coverage**. The probe
evoked one spike per neuron, conserved total K to its recorded floating-point
precision and raised glial intracellular K from 140 to 140.000071916 mM.
Glial voltage moved from -100 to -99.653781214 mV. GABA reduced the monitored
neuronal voltage by 0.007476803 mV relative to its control. The short glutamate
calcium difference was only 1.2744e-14 mM: this is an early numerical response,
not evidence of a physiologically substantial calcium wave. The probe took
16.72 seconds including assembly/compilation and both controls.

## Edge cases, corrections and next tests

Tests reject overlapping cables, wrong E/I maps, duplicate/out-of-range sources,
absent release owners, mismatched clocks/volumes/calcium maps and integer initial
voltages. They exercise disabled release, malformed/negative molecule inputs,
uptake and boundary accounting. Existing pool-ownership checks require every
intracellular pool exactly once.

The replay regression exposed integer initial voltage becoming floating after
one step. The constructor now rejects it before compilation; checkpoint dtype
validation remains strict. Test setup corrections use actual IonInfo pytrees,
native population spike aggregation and complete CableProperty declarations.
Future fixtures must preserve these runtime contracts. The independent mechanism
was rebuilt from its pinned source instead of assuming a previous compiled
library retained its original MOD files.

Next qualification tests should cover measured glial fragments with explicit
spatial/source manifests, coupled time/space refinement, sustained calcium-wave
propagation, myelin electrical dynamics and complete H01 factory reconstruction.
Then qualify sparse learning and measured 4/12/40-cell runtime before all-104
forward/replay/training. Synthetic roles, volumes and glutamate dose here do not
establish human physiology. No long biological waits, full 104-cell simulation
or GPU gate were performed in this phase.
