# Selective sodium recovery diagnostic

Use the factor-1.5 calcium midpoint as control. Its first interval is
26.502158 ms, compared with 33.851739 ms in the human recording. The
tested calcium-removal range has little effect on this first interval.

Test whether slowing transient sodium recovery lengthens the first
interval without changing first-spike onset or peak. In the source NaTs
law, h approaches hInf with time constant hTau. Add a diagnostic factor
only when hInf > h. Keep closing kinetics, hInf, activation kinetics,
channel density, and all other model parameters unchanged. Factor 1
must reproduce the source behavior. Test factor 2 for the intervention.

Before the cell experiment, verify at fixed voltage that the recovery
time doubles, closing time stays fixed, and equilibrium stays fixed.
Use an isolated copy of the mechanism sources and retain the originals.
Record source hashes and the exact patch. Compare the factor-one build
with the saved midpoint control. Require exact time, voltage, and current
prefix equality at fixed dt 0.000625 ms before using it as a control.
If this fails, preserve the failure and diagnose it before proceeding.

Keep the factor-nine mesh, complete command-only input, and 2100 ms
endpoint. Before observing the factor-two cell response, require a
smaller absolute first-interval error than the control. The first onset
must change by no more than 0.1 ms and the first peak by no more than
0.1 mV. Required first and second events must be complete with positive
peaks. Missing observations invalidate this diagnostic.

Retain the first minimum, all later intervals, event counts, and human
residuals even if they worsen. Passing the narrow recovery prediction
does not qualify the whole cell. This new kinetic factor is an inferred
intervention, not a measured human channel parameter. Do not attribute
a human biological mechanism uniquely from a model-only intervention.

Fixed-voltage checks use -80, -60, -40, and 0 mV, factors 1 and 2,
and initial h values 0, 1, and the source equilibrium. Set gbar to zero
so each isolated section retains its prescribed voltage without clamp
error. Compare complete h and m trajectories with the analytic source
exponentials over 2 ms at dt 0.00005 ms. Require absolute gate error
<=1e-8 and unchanged voltage. The activation gate must retain its source
time constant in all cases. Advance all independent sections together
with one continuerun call, without a Python time-step loop.

## Candidate numerical check

Before selecting another recovery factor, compare the factor-two cell
with CVode at absolute tolerance 1e-10. Keep the same mechanism library,
source hashes, factor-nine mesh, calcium factor 1.5, complete input,
initial state, temperature, and 2100 ms endpoint. Retain raw adaptive
records and verify their right-limit representation. Require equal
event counts and peak signs. Every corresponding onset must differ by
at most 0.1 ms, peak by at most 0.1 mV, and duration above -20 mV by at
most 0.01 ms. Retain every difference. A failed or invalid check prevents
numerical qualification of this candidate; do not relax the limits.

For candidate spatial qualification, compare mesh factors 3 and 9 at
fixed dt 0.000625 ms. Reuse the completed factor-nine trace. Hold sodium
factor 2, calcium factor 1.5, source build, input, and endpoint fixed.
Require tripled segment counts, unchanged section geometry and parents,
and the same per-event limits stated above. A pass qualifies this one
successive refinement, not all meshes, inputs, or human responses.
