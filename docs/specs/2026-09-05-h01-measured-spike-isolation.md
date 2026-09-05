# Measured H01 output-event diagnosis

## Fixed system and datum

Use I 5584343344 and E 4157825456, component 0, and contact 8105899.
Keep the pinned source hashes, region map, channel profiles, 10 um maximum
CV length, 0.005 ms step, initial states, and 8 ms observation window.
The soma pulse starts at 2 ms and ends at 5 ms. E receives 1 nA.
The baseline I pulse is 1 nA. These inputs are model interventions, not
measured H01 stimuli. The H01 release supplies no matched voltage datum.

The saved baseline is `.cache/h01/measured-ie-index-fixed.npz`.
Its I soma reaches -29.53 mV. Its declared output contact stays below -80 mV
and emits no event. E synaptic conductance stays zero. All recorded samples
are finite. Unrecorded compartments are not classified.

## Numerical prerequisite

Compare every saved sample from `measured-ie-scan.npz` with the baseline.
Require identical sample times and event arrays. Require voltage agreement
with rtol 1e-10 and atol 1e-9 mV. Require conductance agreement with rtol
1e-10 and atol 1e-12 uS. Reject missing keys, shape mismatch, or nonfinite
values. Check identical source hashes, profiles, contacts, inputs, and mesh.
Only the integrator name may change. Keep both raw outputs.

This checks implementation equivalence at one step and one input. It does
not establish time-step convergence or human physiology.

## First physical split

After numerical equivalence passes, increase only the I pulse to 2 nA.
Keep E input and the measured I-to-E connection fixed. Record each cell's
soma voltage, declared output voltage, event sequence, and conductance.
Do not promote a changed channel law or electrical map in this split.

The specific question is whether this input change produces a positive
I soma excursion and an event at the declared I output site. Report each
answer separately. Retain the full voltage trace; a positive excursion
alone is not proof of a physiological action potential.

- Soma crosses 0 mV, contact emits: test delivery with only the edge removed.
- Soma crosses 0 mV, contact does not emit: record direct voltages along the
  soma-to-contact path before changing its electrical regions.
- Neither occurs: the split does not distinguish insufficient current,
  initiation failure, depolarization block, or attenuation before the soma
  probe. Inspect channel availability and local voltages before another fit.
- Contact emits without a positive soma excursion: soma voltage is not a
  sufficient initiation readout. Locate the first local event on the cable.

The next propagation split must hold the initiating waveform or demonstrated
local event fixed. A region change that also creates the initiating event
cannot by itself isolate propagation failure. The sparse source labels remain
unchanged in any electrical-map counterfactual.

## Delivery and qualification

If I emits, compare connected and disconnected responses with identical
inputs and initial states. Require identical I events, a delayed conductance
at the measured E contact only in the connected case, and a direct E voltage
difference consistent with the local driving potential. Compare the trace
before delivery as well as after delivery. Repeat at half dt before treating
the effect as numerically robust. Distinguish the borrowed synaptic dynamics
from measured contact anatomy. Human response qualification remains separate.

## Sodium closing-time intervention

The 2 nA split does not produce a positive recorded soma excursion or an
output event. E remains exactly unchanged. Return to the 1 nA baseline and
use the existing `h01_i_source_closing_circuit` diagnostic runner. Restore
only the NaTg inactivation closing-time factor from 0.15 to 1 in I. Keep
opening time, equilibrium slope, density, other channels, and the electrical
map fixed. This is an intervention on a channel law, not a fitted candidate.

The narrow claim is that the candidate closing-time change prevents a
positive soma excursion under this fixed input. A positive excursion only
after restoration supports that claim in this model context. An absent
positive excursion contradicts this specific rescue prediction. Neither
outcome alone proves where an action potential starts or why it fails to
reach the measured contact. Preserve the full soma and contact traces, verify
the unchanged pre-pulse response, and report contact emission separately.
The source law was borrowed from a different human-cell model; restoring it
does not establish H01 physiological accuracy. Do not promote the override.

## Contact delivery after the longer observation

The restored-closing all-CV run emits at the measured I contact at 8.355 ms.
The contact peaks at +28.25 mV at 8.465 ms. Its first 8 ms exactly match the
circuit record. The 8 ms window was insufficient to test arrival in this
condition. No electrical-region change is justified by that truncated record.

Keep the restored closing-time law and both 1 nA soma pulses. Run the measured
pair for 20 ms with the I-to-E edge enabled and disabled. Add a voltage probe
at the E receptor location, already used by the conductance probe. Verify on
the fixture that this probe selects the intended local CV and does not alter
the established soma traces. Retain that voltage in exported arrays.

Require identical I traces and events in both controls. E conductance must
stay zero without the edge and start only after the declared 0.5 ms delay
with the edge. E voltage must match before delivery. Use the local E voltage
relative to -80 mV to determine the initial current direction. Do not label
the effect hyperpolarizing merely because the source cell is inhibitory.
These controls test the measured contact with borrowed dynamics. The restored
law is still diagnostic, and neither E nor I has human-response qualification.
