# Regional mesh diagnosis

The candidate's factor-9 to factor-27 late interval changes by 1.844121 ms.
The tested CVode tolerance effect is much smaller. Isolate mesh changes in
the soma, axon, and dendrites at 0.27 nA. Keep all model parameters fixed.

Add an optional positive odd `--unselected-nseg-factor` with default 1 to
the diagnostic NEURON driver. Selected sections retain `--nseg-factor`.
Unselected sections use the new factor. Default calls must retain their
existing mesh. Test actual section allocation for all regions, including
both basal and apical dendrites, and invalid factors before simulation.

For each regional run, selected sections have source nseg times 27 and
all other sections have source nseg times 9. Compare with the saved all-9
and all-27 runs. Verify geometry, parameter, and per-section count invariants.
Retain all events, early intervals, and the selected late positive pair.

For a region to reproduce the full-refinement late-interval effect in this
diagnostic, require its selected late interval to be within 0.25 ms of the
all-27 interval. Missing positive pairs invalidate that comparison. Preserve
every region's signed change from all-9. This is a numerical sufficiency
screen, not proof of a unique biological cause. Regional effects need not
add in a coupled nonlinear cell. No production model is promoted.

## Axon-focused reference comparison

Run axon factor 81 with other regions at factor 9. Compare against the
saved all-81 response at the same high input and CVode tolerance. Verify
actual section counts and physical properties. Before the run, require
equal complete-event counts, matching positive/negative event classification,
each ordinal onset difference at most 0.1 ms, and the selected late interval
difference at most 0.01 ms. Missing late pairs invalidate this check.
These engineering limits test whether the smaller mesh reproduces this
reference. They do not establish that the all-81 reference is converged.
Retain peak and duration differences and every unmatched event separately.

## Successive axonal refinements

Run axon factors 243 and 729, retaining factor 9 elsewhere, the same
0.27 nA input, and CVode tolerance 1e-10. Compare focused 81 to 243 and
243 to 729 separately. Require equal complete-event counts, identical
positive/negative classification, every ordinal onset change at most
0.1 ms, and selected late-interval change at most 0.01 ms for each pair.
Missing late pairs invalidate the decision. Keep failed comparisons.

Verify exact non-axonal section counts, tripled axonal counts, and physical
geometry and parameter invariants. Retain all events, peak and duration
differences. A passing pair supports only this numerical observation set
and parameter/input condition; physiological qualification remains separate.

Factor 243 to 729 fails the 0.1 ms onset limit, although the selected late
interval passes. Next run axon factor 2187 with factor 9 elsewhere and
compare 729 to 2187 using the same count, classification, onset, and
late-interval limits. Do not round 0.113263 ms down to a passing 0.1 ms.

Before factor-2187 output exists, make a separate numerical prediction
for event 40. The preceding shifts are 1.016766 and 0.113263 ms.
If the leading spatial error scales with squared compartment length,
the next shift is about 0.012585 ms. Check the predicted crossing time
against the actual value with absolute error at most 0.005 ms. Missing
event 40 invalidates this prediction test. The saved prediction is
`docs/evidence/h01-pv-axon2187-prediction.json`. Agreement would support
this local error-scaling account only; it cannot replace the full-event
convergence criteria or establish biological validity.
