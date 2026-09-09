# H01 current observations

## Result

The I traces locate a large axial outflow after the trough.
They do not identify the change needed to reproduce human timing.
The saved B3 traces lack the currents needed for an equivalent explanation.

This review analyzed existing data. It ran no simulation.
The [data record](h01-causal-current-observations-2026-09-09.json) contains file hashes, settings, states, and all sampled current terms.

## I: current after the trough

Use the donor finalist `e-kv3-close2` at 0.19 and 0.23 nA.
Both runs use command-only input, CVode tolerance 1e-10, and ninefold spatial refinement.
The 0.23 nA recording was already evaluated. This analysis supplies no independent validation.

The boundary is the recorded soma compartment, with area 154.666 µm².
Sample 6 ms after the first, second, and penultimate troughs.
Each sample precedes the next threshold, defined by the existing 10 V/s criterion.
Currents below are positive inward. Storage current is positive when soma charge increases.

| Input (nA) | Spike | Sodium availability | Axial current (nA) | Storage current (nA) | Trough to next threshold (ms) |
| --- | --- | --- | --- | --- | --- |
| 0.19 | 1 | 0.9857 | −0.18058 | 0.00198 | 31.44 |
| 0.19 | 2 | 0.9868 | −0.18043 | 0.00199 | 35.80 |
| 0.19 | 13 | 0.9881 | −0.18075 | 0.00180 | 80.84 |
| 0.23 | 1 | 0.9809 | −0.21842 | 0.00256 | 17.60 |
| 0.23 | 2 | 0.9825 | −0.21830 | 0.00254 | 18.85 |
| 0.23 | 25 | 0.9860 | −0.21937 | 0.00204 | 40.28 |

All eight recorded neighbors have lower voltage than the soma at these samples.
Their voltage differences and axial resistances give the reported outflow independently of the current residual.
Axial outflow equals 94.9–95.4% of the applied current.
This ratio is a local current comparison, not a whole-cell energy fraction.

At the first 0.19 nA sample, leak and Kv3 carry 0.00686 and 0.00610 nA outward.
NaTg, Ca_LVA, and Ih carry 0.00068, 0.00238, and 0.00262 nA inward.
The other ionic terms are smaller at this sample.
The signed sum leaves 0.00198 nA to increase soma charge.

Thus, recovered sodium availability does not imply rapid voltage recovery.
The soma sends most applied current into connected cable at these times.
The downstream storage and ionic paths require their own measurements.
A large axial current alone does not establish excessive cable load relative to the human cell.

Axonal calcium and SK activation increase between the early and late samples.
At 0.23 nA, calcium rises from 0.0001202 to 0.0001987 mM.
The SK gate rises from 0.00219 to 0.02417.
The next-threshold delay also increases, but these joint changes do not isolate a cause.

## I: correction to the earlier cause test

The Stage X SK arm was not a single-change comparison against its stated candidate.
The [baseline metadata](h01-i-energetic/r-candidate-019.json) apply somatic NaTg at 1.1 times its source density.
The [SK-arm metadata](h01-i-energetic/x-sk-axon-019.json) replace that intervention with axonal SK at zero.
The [runner](h01_pv_neuron_reference.py) applies the selected conductance scale; these records specify no additional regional scale.
Thus, this comparison also returns somatic NaTg density to its source value.

Both Stage X configurations use Kv3 closing factor 0.5.
The retained finalist uses factor 2.0.
The 57/138-spike result therefore does not isolate SK in the finalist.
It remains an observation of the earlier configuration with two changes.

The somatic Kv3 removal arm retains the baseline sodium scale.
Its depolarization block supports a Kv3 role under that earlier configuration.
It does not establish necessity under every parameter set.

The previous account relied on the intervention label without checking the complete parameter difference.
Future cause tests must compare all applied settings before assigning an effect to one change.

## E: missing B3 observations

All four inspected B3 traces share candidate hash `e4825c83bdc7980a469b5501b2a54c68e680f1cb52a2055d36bcf4fc7b70e564`.
They retain time, soma voltage, axon voltage, applied current, and raw sample indices.
They have no channel currents, calcium, gate states, or axial geometry record.
The metadata explicitly set `record_soma_currents` to false.

The [older current manifest](h01-e-energetic-manifest.json) uses sodium factor 1.3 and calcium decay factor 1.33.
B3 uses 0.9 and 1.0, with added axonal NaTs and reduced somatic SK.
The older sodium and axial budgets cannot be assigned to B3.

The count error also changes with input: +3, +3, 0, and −1 spikes.
These values correspond to 200, 250, 310, and 350 pA.
One constant count offset cannot describe this response.
The [gain tests](h01-e-gain/stage-close-decision.json) identify rejected doses, not a complete physical explanation.

## Tests for the missing links

These tests are specified, not executed.
They do not change candidate status or reopen closed fitting campaigns.

**I: isolate the SK effect.** Compare the finalist with an otherwise identical candidate that sets axonal SK density to zero.
Retain somatic NaTg factor 1.1 and Kv3 closing factor 2.0.
Use the established 0.19 and 0.27 nA inputs.
Record local axonal currents and calcium, plus the existing soma and axial probes.

If SK causes the growing delay, its removal must reduce outward axonal charge before the delayed threshold and shorten that delay.
Reject that explanation if the current changes but the delay does not shorten beyond numerical uncertainty.
Loss of spikes or depolarization block would also reject the proposed timing repair.
An effect on timing alone would not establish a valid human waveform.

**E: measure B3 current paths.** Rerun unchanged B3 at 250 and 310 pA with the existing soma-current recording option.
Preserve candidate values, input bias, mesh, tolerance, and recording convention.
First reproduce the saved voltage response within the registered numerical limits.
Then compare the first, second, and late spikes, including their intervening recovery windows.

Separate sodium inflow, potassium outflow, axial exchange, and charge storage.
Check somatic calcium, Ih activation, and Im activation where the current runner records them.
If a proposed current has the wrong sign or occurs after the response change, reject that proposed direct path.
Current measurements must precede selection of a parameter intervention.

The existing runner does not record every channel gate or a complete axonal current budget.
Any explanation that requires those observations needs additional instrumentation and a separate implementation specification.

## Checks and limits

All arrays in the two I traces are finite, and their times increase strictly.
The six selected samples precede the next threshold.
Their current-balance residuals are below 1e-15 nA.
This checks recorded-current consistency; it does not establish solution accuracy.

The analysis uses linear interpolation on the recorded time grid.
It uses the existing [landmark and current functions](h01_spike_cycle_energetics.py).
No matching human current measurements are available in these records.
The results explain selected model states, not the corresponding human mechanisms.
