# Human PV calibration evidence

## Scope

Constrain the transferred inhibitory model with released human recordings.
The source cell is putative PV. This is not proof of PV protein expression.
Keep source-model reproduction separate from physiological calibration.
Do not tune biological parameters to compensate for an unresolved mesh error.

## Data and datum

Use the pinned ModelDB 267587 L5PV `active.pkl` and `passive.pkl` exports.
Record source hashes. Keep original time and voltage values.
The current step runs from 270 to 1270 ms.
The active steps are 0.19, 0.23, and 0.27 nA.
The hyperpolarizing steps are -0.11 and -0.05 nA.
Use the 200-270 ms window to report baseline behavior for each recording.

Record each spike's sample peak and crossing times at -20 mV.
Interpolate only the threshold crossing between adjacent samples.
Call the crossing interval “time above -20 mV.” It is not AP half-width.
Retain complete direct traces and sample-wise residuals for all comparisons.
Do not align spikes or shift measured voltage to improve a score.

Within-trace baseline fluctuation is not between-cell uncertainty.
The source optimizer's `std` values are objective weights, not measured uncertainty.
Do not use either quantity as an automatic biological acceptance band.

## Prospective fitting boundary

Reserve the 0.23 nA trace from future parameter fitting.
Use the two other active traces and the hyperpolarizing traces as candidate
calibration data. Record which data each fitted parameter used.
All released traces have already been inspected. The reserved trace is not
an untouched external validation set. State this limit in every validation claim.

## Nested checks

1. Resolve the main numerical error with fixed physical parameters.
2. Check baseline, hyperpolarizing deflection, sag, and recovery.
3. Check individual spike voltage, duration, onset, and recovery.
4. Check the full train and the reserved current step.
5. Transfer only qualified dynamics to the E/I circuit.

The model must explain the direct response, not only spike count.
If a baseline correction requires an inferred input or state, name that assumption
and test it. Do not call it a measured holding current.
Biological parameter bounds must have a source or an explicit inferred status.
Acceptance tolerances and numerical error budgets remain to be established.
This specification does not claim that the current model passes validation.

## Acquisition provenance

Compare released calibration traces with Allen NWB file 618112937.
Check the pipeline version before converting stored values to SI units.
Allen pipeline 1.0 stores SI values despite obsolete conversion attributes.
Check all samples after the 750 ms crop and the -14 mV junction correction.
Read each sweep's measured bias current separately from its stimulus waveform.
Do not refit the reserved 0.23 nA trace during this provenance check.
Keep raw NWB data in the cache. Save hashes and the comparison in evidence.

First add the measured 0.031445374082395006 nA bias to the 0.19 nA
reference. Apply it from model time zero through the whole simulation.
Keep initial voltage, conductances, mesh factor 9, and CVode tolerance fixed.
Record bias and step current separately. Check their sum.
Compare with a zero-bias control from the same driver.
This tests bias alone. It does not reproduce the full acquisition history.

## Bounded conductance sensitivity

Spike peaks remain near 44 mV across completed spatial refinements, while
recorded human peaks are near 17 mV. Test separate 10 percent reductions in
NaTg, Kv3_1, and SK conductance on the published model's soma and axon.
Hold all other parameters, initial state, mesh, and stimulus fixed.
Use mesh factor 9 and CVode tolerance 1e-10 for these diagnostic interventions.
Report direct voltage-change RSS, individual spike peaks, and spike times.
This is a conditional sensitivity result, not uncertainty propagation or fitting.
Do not promote a changed model until numerical and physiological checks pass.

Next isolate somatic transient sodium from axonal transient sodium.
The previous intervention changed both regions and removed spikes.
Test soma-only conductance factors 0.25, 0.10, and 0.05, leaving axon unchanged.
These densities remain within the source optimizer's somatic 0-0.5 S/cm2 bounds.
The factors are diagnostic choices, not measured human parameter values.
Record soma and axon voltage together. Check retained initiation, soma peaks,
the full train, and calcium response before considering a fitted candidate.

The first soma-only reductions removed both soma and axon spikes at 0.19 nA.
Next test factors 0.75 and 0.90 under the same conditions. These bracket a
smaller reduction. Check whether peak voltage changes before spike generation fails.
Do not infer a measured sodium density from this diagnostic bracket.

Test soma-only Kv3_1 factors 2 and 4 with sodium unchanged.
The resulting densities are 0.857697 and 1.715394 S/cm2.
Both lie within the source optimizer's 0-3 S/cm2 bounds.
Use the zero-bias source reference, mesh factor 9, initial -80 mV,
0.19 nA step, and CVode tolerance 1e-10.
Check individual peaks, time above -20 mV, and the full train.
These are diagnostic parameter choices. They are not measured densities.

Observe somatic sodium gates, Kv3 activation, and each channel current during
the unchanged source response. Keep NEURON outward-positive current signs.
Check per-channel currents against the ion-current sums and confirm that
adding probes leaves the voltage response unchanged. Use the observations
to identify sustained inward current before selecting a kinetic intervention.

## Waveform candidates from the causal result

The closure-only factor 0.5 shortens the first spike on two spatial meshes.
It still leaves the first peak too high and the duration too long.
Test closure factors 0.25 and 0.10 with recovery factor 1, keeping other
parameters at the source values. Use 0.19 nA and mesh factor 9 initially.
These factors are inferred diagnostic candidates, not human-measured kinetics.
Retain all events. Compare first peak and time above -20 mV with the recorded
19.5625 mV and 0.27339465 ms. A candidate must improve both absolute errors
and preserve a complete spike to justify further waveform calibration.
This screening rule is not physiological validation. Check later events,
the 0.27 nA calibration trace, numerical robustness, and the reserved 0.23 nA
trace before accepting a final model. No parameter is promoted by this screen.

Next test closure factor 0.18 at both calibration currents, 0.19 and 0.27 nA.
This inferred factor lies between the completed 0.10 and 0.25 cases.
Keep recovery factor 1 and all other parameters fixed. Retain each event.
Check positive spike peaks, repeated spiking, event durations, and onset times.
Do not accept a first-event error reduction as proof of a valid train.

Test SK conductance factor 2 in soma and axon on closure factor 0.18.
Keep recovery factor 1 and other parameters unchanged. Use both calibration
currents. The doubled densities remain within the source optimizer's 0-1
S/cm2 bounds. This is an inferred candidate, not a measured density.
Prediction: later interspike intervals increase relative to the 0.18 candidate.
Retain every interval and first-event shape. A lower count alone does not pass.
Report any loss of spikes or remaining onset error. Do not infer that SK is
the unique cause of adaptation from this conductance intervention.

Split the doubled SK intervention into soma-only and axon-only changes at
0.19 nA with closure factor 0.18. Compare with the existing unchanged and
both-region cases. Retain absolute rising crossings and all interspike intervals.
If only one regional intervention reproduces the first-crossing delay within
0.1 ms of the both-region result, identify that region as sufficient for this
delay under these conditions. Otherwise leave the regional attribution unresolved.
This threshold is a diagnostic distinction, not a human tolerance. Do not add
regional effects as independent errors or assume sufficiency proves uniqueness.

Test axonal calcium removal time 1000 ms on the 0.18 closure-factor candidate,
with original SK density. The source optimizer allows removal times 20-1000 ms.
Hold initial calcium, calcium entry scaling, and all other parameters fixed.
Use both calibration currents. Record axonal calcium, SK gate, and SK current.
Prediction: slower removal increases retained axonal calcium and delays later
crossings, with a smaller first-crossing shift than doubling axonal SK density.
Evaluate that onset distinction at 0.19 nA against the existing regional split.
If the initial delay is not smaller, reject this proposed separation at that input.
Do not claim a full causal pathway from longer intervals alone.
