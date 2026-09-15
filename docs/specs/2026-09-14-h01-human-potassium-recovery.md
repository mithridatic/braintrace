# Human-only potassium recovery estimation

Approved parent scope: human-only mechanism replacement, with all six original
acceptance requirements retained. This experiment estimates an effective recovery
response from the acquired human measurements, before any neuronal integration.

## Observable and source boundary

Use only the 13 human rows in the pinned Figure 4 workbook (source checkpoint
745e525b). The observable is `amp1torec_1` through `amp1torec_10`, the current
amplitude relative to its initial amplitude, versus `recdelay_1` through
`recdelay_10` in ms. Require protocol `K_IKrecv2_1_h0_DA_0` (paper Fig. 4F,
recovery at commanded -80 mV, 34 C). The source plotting script applies this
protocol filter without a recovery goodness-of-fit filter. Retain all its
available human curves and all finite observations, without clipping, outlier
exclusion, amplitude renormalization or subtraction of a fitted baseline.
Keep missing-protocol rows in an explicit exclusion receipt. Reject incomplete
time/amplitude pairs, duplicate identities, nonhuman labels, nonfinite data and
non-increasing or negative delays. A completely absent pair may be skipped with
its index recorded. Require at least six complete pairs per included recording.

Three additional workbook amplitude columns have suffixes ending `.1`; they
appear to duplicate indices 8, 9 and 10. Require exact equality to the canonical
field (including paired absence) before excluding them as duplicate columns.
They cannot add independent observations or statistical weight.

The dictionary's inactivation `ms-1` labels conflict with the named time
constants, the paper's exponential current model and figures, and the source
fitting functions using `exp(-t/tau)` with t in ms. Interpret those constants as
ms for future work, explicitly as a source-reconciliation inference; preserve
the original bytes. This run uses recovery delays and dimensionless amplitude
ratios, whose definitions are explicit, and does not use those inactivation
fields or the ambiguously labelled `rec_ss` field as targets.

## Model and computation

Fit the effective current-amplitude recovery curve

R(t) = r0 + (rinf-r0) [f (1-exp(-t/tfast)) + (1-f) (1-exp(-t/tslow))].

Use r0 in [0,1], recovery fraction q in [0,1] with
rinf=r0+(1-r0)*q, f in [0,1], tfast in [0.1,1000] ms and
tslow in [1000,100000] ms. The 1000-ms division separates exponential labels
and follows the released recovery fitter's cutoff; bounds are numerical modeling
choices, not independently measured human properties. Time constants optimize in
log space. Fit a three-parameter single-exponential alternative under the same
amplitude constraints, with tau in [0.1,100000] ms. Use equal weight per recording
and within each recording; deterministic least squares, at most 2000 evaluations
per start, three fixed starts (fast 50/200/700 ms, slow 1500/4000/15000 ms;
single tau 200/1000/4000 ms). Initial r0=.15, q=.7, f=.5 for each start.
No random search, parameter-bound expansion or extra retry after inspecting errors.

This is vectorized closed-form observation algebra, not a repeated neuronal
simulation. It needs no GPU, remote job or Python-driven neuronal time loop.
The complete CPU fitting and cross-validation process has a 120-second wall cap.
Retain every start's outcome, best parameters, residual samples, evaluation count,
bound proximity and Jacobian condition diagnostic. An unconverged optimizer
cannot produce a selected candidate. No production parameters change.

## Validation and decisions

Before fitting, assign recording groups by the first three dot-separated filename
components: H21.29.190 and H21.29.191. These are filename groups, not confirmed
independent donors. Fit on each group and predict the other group without using
its responses in that fit. All inputs were previously acquired and descriptive
plots inspected; call this grouped cross-validation, not a sealed blind holdout.
Then estimate one candidate on all eligible human recordings, retaining both
alternative fits and their direct predictions.

Report per-record RMSE, signed residuals at all sampled delays, early
(<=300 ms), intermediate (>300 to <=2000 ms) and late (>2000 ms) errors, and
common-scale plots for every recording. Favor the two-exponential family for
further protocol modeling only if it improves equally record-weighted grouped
validation RMSE over the single exponential and neither group's RMSE worsens.
This is a comparative model-selection rule, not physiological acceptance.
Retain rejected results. Do not change the rule after seeing results.

A current-amplitude recovery constant is not automatically a channel-gate rate.
Its mapping depends on conditioning pulse, initial state, activation, measurement
window and membrane/control errors. Original experimental current traces and the
complete experimental command still must be obtained or independently verified
before claiming waveform or gate qualification. This run cannot promote donor
accuracy, type coverage, anatomy transfer, timestep or driven-window scores.

Test analytic limits, monotonicity, single-exponential degeneration, species and
protocol selection, missing pairs, duplicate-column handling, group separation,
known synthetic parameter recovery and rejection of invalid optimizer inputs.
Use meaningful sibling tests and inspect the generated direct-observation plots.
