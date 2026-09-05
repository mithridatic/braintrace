# Somatic calcium-conductance split

The slope-5 candidate retains inward Ca_LVA current at the first post-spike
minimum. Test whether half the somatic Ca_LVA conductance makes that minimum
more negative. Keep axonal calcium conductance, SK, and calcium removal fixed.
Retain sodium slope 5 mV, closure factor 0.15, recovery factor 1, and soma
NaTg density factor 1.10. Use mesh factor 9, CVode tolerance 1e-10, and the
0.19 and 0.27 nA calibration inputs. This is an inferred conductance change.

Predeclare the directional test at each input: the first post-spike minimum
must be at least 1 mV more negative than the unchanged-conductance candidate,
and the first two events must retain positive peaks. Define the minimum
between the first sampled peak and the second upward -20 mV crossing.
If the second positive spike is absent, the directional test is invalid,
not a success. Retain peak, duration, onset, and all later events separately.
Also compare the minimum's time from peak with the human recording.

This tests a conductance intervention in the coupled model. Calcium entry
can alter calcium concentration, reversal voltage, and SK activation; do
not attribute the result exclusively to a removed instantaneous current.
Do not read the reserved input or promote a candidate from this split.

The new driver factor defaults to 1, permits 0, and rejects negative or
nonfinite values. Record it explicitly. Older control metadata predates
this option and represents unchanged source conductance.

## Compare the fall at matched voltages

On existing human, control, and half-conductance traces, report first downward
crossings of -20, -40, -50, -60, -65, -70, and -75 mV after the first peak
and before the second upward -20 mV crossing. Interpolate only adjacent
samples. Retain absolute crossing time, time from peak, missing crossings,
and repeated-crossing counts. Do not shift the raw traces.
Use these observations to distinguish changed minimum depth from a delay
in reaching the same voltage. A missing low-voltage crossing is a direct
response mismatch, not a reason to extrapolate a crossing time.
