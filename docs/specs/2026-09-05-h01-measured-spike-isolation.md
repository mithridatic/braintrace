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
