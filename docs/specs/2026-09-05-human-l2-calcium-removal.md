# Layer-2 calcium-removal split

The source calcium law is

`dc/dt = -10000 * ica * gamma / (2 * F * depth) - (c - c_rest) / tau`.

At fixed calcium above rest and fixed calcium current, increasing tau
reduces removal. It does not change the entry multiplier gamma or the
SK density. In the full cell, calcium also affects calcium reversal;
therefore the intervention must not be described as an isolated SK test.

Question: Can slower removal lengthen the short recovery intervals
without materially deepening the already too-negative first minimum?

Use the retrieved source fit and complete command-only waveform. Double
only somatic decay_CaDynamics from 494.01955262344603 ms. Keep gamma,
all channel densities, geometry, temperature, and initial-state rule fixed.
Use the mesh established by the pending spatial check; if that check
fails, diagnose numerical sensitivity before selecting the comparison mesh.
Do not launch this intervention before the mesh choice is documented.

Before observing the intervention, declare the diagnostic prediction:
the first three onset intervals must each have smaller absolute error
against the human intervals than the corresponding control interval.
The first interspike minimum must not become more negative by more than
0.1 mV, and the first sampled peak must change by no more than 0.1 mV.
All conditions must hold to support this prediction. Missing required
events invalidate the comparison. Retain all event counts and times;
a smaller count alone is not a pass.

The required four model events must have positive sampled peaks.
An excursion above -20 mV alone is not sufficient to label it a spike
for this comparison. Human reference intervals must be finite and
positive. Missing these conditions invalidates the prediction test.

These limits test the proposed separation of recovery duration from
rapid return. They do not establish biological parameter values or
physiological qualification. No result may be called a pure removal-rate
measurement in the coupled cell. Repeat relevant numerical checks if
the intervention changes the firing regime.

Implementation must preserve the source fit object, change exactly one
matching soma row, and reject missing or duplicate target rows, invalid
source values, and nonpositive or nonfinite factors. Default factor 1
must preserve all source values. Record the source and applied decay.

The exact observation-prefix check permits a 2100 ms endpoint for this
main-pulse experiment. Keep the entire command vector and pre-pulse
history. Compare with the control only on the retained observation
window. The control may have a longer post-pulse tail. Record this
endpoint difference explicitly; it is not a change to the prior input.

Mesh selection: the factor-three to factor-nine source comparison passes
all declared event and structural limits. Use factor nine, dt 0.000625 ms,
and the command-only input. Control is h01-l2-source-sweep50-space9.
The intervention uses a 2100 ms endpoint and decay factor two. Temporal
qualification was obtained on the source mesh; finer-mesh time/solver
agreement and intervention-specific numerical checks remain separate.

The fixed-step intervention supports the diagnostic prediction but
changes the event count from seven to four. Compare this same doubled
decay candidate with factor-nine CVode at tolerance 1e-10. Keep the
2100 ms endpoint and all model/input settings fixed. Require the same
event count and peak-sign classes, every onset difference <=0.1 ms,
every sampled peak difference <=0.1 mV, and every duration difference
<=0.01 ms. Preserve any failure and do not infer intervention stability
from the source-model solver comparison alone.
