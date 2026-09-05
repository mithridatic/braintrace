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

## Bounded conductance sensitivity

Spike peaks remain near 44 mV across completed spatial refinements, while
recorded human peaks are near 17 mV. Test separate 10 percent reductions in
NaTg, Kv3_1, and SK conductance on the published model's soma and axon.
Hold all other parameters, initial state, mesh, and stimulus fixed.
Use mesh factor 9 and CVode tolerance 1e-10 for these diagnostic interventions.
Report direct voltage-change RSS, individual spike peaks, and spike times.
This is a conditional sensitivity result, not uncertainty propagation or fitting.
Do not promote a changed model until numerical and physiological checks pass.
