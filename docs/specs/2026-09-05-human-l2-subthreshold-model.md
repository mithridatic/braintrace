# Fixed-parameter subthreshold comparison

Run the calcium-midpoint candidate on calibration sweep 43 (110 pA).
Keep source sodium density and recovery, calcium factor 1.5, mesh factor
9, initial state, and temperature unchanged. Use CVode tolerance 1e-10
and stop at 2100 ms. Retain the full source command prehistory. Initially
use command only, matching the prior model convention; report the
recorded bias separately and do not claim exact total-current matching.

Record corrected human and model voltages at 1019, 1025, 1030, 1040,
1070, 1120, 1520, 2019, 2025, 2030, 2050, and 2099 ms. Interpolate
model voltage onto those exact times and identify that operation. Keep
all direct signed errors. Require no complete -20 mV model events during
the pulse as a necessary behavior check. This check alone does not
qualify the subthreshold waveform. Do not invent a human error tolerance.

Distinguish baseline offset from stimulus-driven deflection by reporting
both absolute voltage and change from the 1019 ms sample. A persistent
deflection error directs attention to subthreshold currents and load
before further spike-channel fitting. This comparison does not uniquely
identify a conductance or prove a passive mechanism. Reserved sweep 53
remains unavailable to this diagnostic. Allow only sweeps 43 and 50 in
the driver until another protocol is specified.

## Recorded-bias sufficiency split

Repeat sweep 43 with its recorded constant bias added to the complete
command. Hold every model and solver setting fixed. Retain the same
12 observation times. Test whether this input correction alone restores
both observed directions: voltage at 2019 ms is lower than at 1120 ms,
and voltage at 2099 ms is below the 1019 ms starting sample. Require no
complete pulse spikes. These are necessary shape conditions, not full
waveform tolerances. Report each condition and all voltage residuals.
If either direction remains wrong, reject bias as a sufficient repair.

## Subthreshold tolerance refinement

Compare the bias-included control at CVode absolute tolerances 1e-10
and 1e-11. Keep all physical parameters, mesh, input, and initial state
fixed. Require absolute voltage differences no greater than 0.01 mV
at each of the 12 predefined times, no complete pulse spikes in either
run, and unchanged truth values for the two response-direction checks.
Retain every signed difference. This is a numerical threshold, not a
human physiological acceptance interval. Verify raw mapping, endpoint,
and applied input before interpreting the comparison.

For spatial refinement, compare mesh factors 3 and 9 at CVode tolerance
1e-10 using the same bias-included control. Require tripled segment
counts, unchanged section geometry, parents, parameters, and input.
Apply the same 0.01 mV limit independently at each selected time and
require unchanged response-direction results and no pulse spikes.
This qualifies one successive refinement at these observations only.
