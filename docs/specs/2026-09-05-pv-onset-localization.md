# Localize the onset mismatch

The closure-factor 0.18 candidate has a late first rising crossing at low input,
but nearly matches that crossing at higher input. Determine where the error
accumulates before changing another conductance.

Use the original stimulus clock and corrected human voltages at 0.19 and
0.27 nA. Do not read the reserved 0.23 nA trace.
Record the first upward crossings after pulse onset at -80, -75, -70, -65,
-60, -55, -50, -45, -40, -30, and -20 mV before the first spike peak.
Interpolate crossing times only between adjacent samples.
Report signed model-minus-human crossing errors and elapsed time between
successive voltage levels. Preserve the direct sampled traces.

This is localization, not a causal intervention. A crossing error can be
caused by conductance, capacitance, geometry, or initial state. Do not infer
a unique parameter from the voltage trajectory alone.
Report missing crossings and noisy repeated crossings explicitly. Restrict
to the first upward crossing by definition; do not select a crossing to improve fit.
Use the result to define a subsequent discriminating test at the relevant
voltage range, with paired current and voltage observations in the model.

## Current accounting at onset

Use the existing local charge-balance probe on the same candidate at both
currents. Keep all model parameters fixed. Require exact equality of the
recorded time and soma voltage with the earlier candidate runs.
Calculate axial current from neighbour voltage and axial resistance.
Check applied inward plus axial inward minus ionic outward minus capacitive
current. Require an absolute residual below 1e-5 nA outside initialization
and stimulus discontinuities.

At each first upward voltage crossing, retain each channel current, total
axial current, and capacitive current. Compare states at the same voltage.
These are current accounts, not selective interventions. A small current at
one soma segment does not exclude an effect through the axon or dendrites.
Use these observations to select a later intervention; do not claim that
the largest current uniquely causes the human-model mismatch.

## Regional sodium split

At 0.19 nA, increase NaTg conductance by factor 1.10 in the soma only,
axon only, and both regions. Use closure factor 0.18, recovery factor 1,
original calcium removal and SK density, mesh factor 9, and CVode tolerance
1e-10. Keep the stimulus, temperature, geometry, and initial state fixed.
The conductance change is inferred, not measured in the human cell.

Test the claim that added sodium conductance advances the first upward
-20 mV crossing by at least 1 ms. Report a yes/no outcome for each region.
If one region reproduces the both-region advance within 0.1 ms and the
other does not, that region is sufficient for that observed advance under
these conditions. Otherwise regional sufficiency remains unresolved.
These are diagnostic decision thresholds, not biological uncertainty.

Preserve all complete events and raw traces. Report the first peak and
duration above -20 mV against the same human targets. An onset improvement
does not pass the waveform screen if either absolute shape error increases.
Do not infer independence or add the regional changes. Do not use the
reserved 0.23 nA trace. A candidate that passes this split still needs
multi-current and numerical checks before promotion.

## Combined candidate after the sodium split

Test soma NaTg factor 1.10 with closure factor 0.15 and recovery factor 1
at 0.19 and 0.27 nA. Retain source SK density and calcium removal.
The lower closure factor tests whether faster closure can offset the peak
increase from sodium density. These parameters are inferred candidates.
Use the same mesh factor 9 and CVode tolerance 1e-10.

Compare every first-event error with the closure-0.18, source-density
candidate at the same input. Report onset, peak, and duration separately.
Declare dominance only if all three absolute errors do not increase at
both inputs, with at least one strict improvement. Otherwise report the
tradeoff without promoting the candidate. This is an optimization screen,
not a biological tolerance or a causal isolation test.
Retain every later event and interval. No aggregate score can override a
failed direct observation. Do not read the reserved current trace.

## Recovery rescue split

The combined candidate stops making positive spikes early at 0.27 nA.
Change only sodium recovery factor from 1 to 0.5. Keep closure factor
0.15, soma NaTg factor 1.10, source SK and calcium parameters, mesh factor
9, and CVode tolerance 1e-10. Run both calibration currents.

The rescue claim is true under this test if the 0.27 nA intervention has
at least two complete positive-peak spikes with upward crossings in
500-1270 ms, while the control has none. Otherwise reject this rescue claim.
Retain every event, interval, and first-event error. This is a diagnostic
rescue criterion, not qualification of the sustained human spike train.
The intervention can establish a conditional effect of recovery speed;
it cannot establish a unique mediator or a human recovery constant.
The piecewise gate implementation has a prior fixed-voltage analytic check.
Any successful rescue still needs numerical refinement before promotion.

Test the opposite direction with recovery factor 2 at both inputs.
Keep the same combined candidate and use the same late-spike rescue rule.
This tests a longer period of reduced sodium availability, not a measured
human recovery rate. Keep the faster-recovery rejection unchanged.
If the higher-input run restores late positive spikes, repeat that run and
its recovery-factor-1 control at spatial factor 27. Require the same binary
rescue decision on both meshes. Report event-time differences separately;
agreement of the binary decision is not full numerical convergence.

## Check the firing failure before further calibration

Independently of the failed recovery rescues, repeat the combined candidate
at 0.27 nA with spatial factor 27 and CVode tolerance 1e-10. Keep recovery
factor 1, closure factor 0.15, and soma NaTg factor 1.10. Compare with the
existing factor-9 run. The observation to check is absence of complete
positive-peak spikes after 500 ms and before pulse end. Retain every event
and the late voltage bounds. If late spikes appear, stop attributing the
failure to channel parameters until the spatial discrepancy is resolved.
If no late spikes appear on either mesh, report two-mesh agreement only;
do not claim full numerical convergence or unique channel causality.

Use the existing raw traces to locate each voltage minimum between the
first three consecutive spikes at both inputs. Define the interval from
the sampled peak to the next upward -20 mV crossing. Keep the first
sample at the minimum if values tie. Report absolute time, voltage, and
elapsed time from the peak. Do not align traces or infer channel causality
from a shallower minimum. This observation will guide the next split.
