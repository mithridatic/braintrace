# Joint human recovery-state candidate

One candidate now carries availability through the first pulse, recovery gap,
second pulse and holding return in all 19 human PAX6 recovery sweeps. The fit
converged in a bounded local CPU run and captures much of the second-pulse and
return behavior. It does not qualify a physiological channel: the first-pulse
response changes across the acquisition sequence, contradicting the candidate's
single shared first-pulse prediction. No six-term H01 score is promoted.

## Source, model and retained observations

Human specimen 840043506, session 840043481, source NWB SHA256:
`30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944`.
The source-bound recovery exports are linked individually in [inputs.json](inputs.json).
Every raw first/second pulse and the first 1000 ms after the second pulse are
retained in [inputs.npz](inputs.npz), including their original timestamps.
The same measured initial baseline is subtracted from all three responses in
each sweep. The post-pulse baseline change is therefore preserved.

Common fitting coordinates use exact sample indices at the source's 25 kHz rate.
Every original timestamp difference was checked against that grid within 1e-8 ms
and retained separately. This accommodates floating-point subtraction at different
onsets, without interpolating a current or changing its sample membership.

The candidate uses two mathematical availability populations. Each population
decays during a +60 mV pulse, recovers during the measured -90 mV gap, and relaxes
during the -20 mV holding return. The second pulse starts from the first pulse's
end state after recovery; the return starts from the second pulse's end state.
A shared off time constant permits the effective current factor to change on
return. Initial availability is assumed to equal the holding asymptote. A single
shared nondecaying pulse term C accounts for current outside the two decaying
terms; it is not assigned an ionic identity.

The exact equations, bounds, starting values and assumptions are in the
[specification](../../../../specs/2026-09-14-h01-human-l1-qualification.md).
These are conditional currents at three amplifier commands, not measured patch
voltages, molecular channel identities or complete voltage-dependent rate laws.
rho and eta are fitted current factors; they are not measured reversal potentials
or isolated activation fractions. Gap current is not predicted by this candidate.

All original samples in pulse phases 10-280 ms and return phases 10-980 ms enter
with equal per-sample weight: 6750 samples per pulse and 24250 return samples per
sweep, **717250 samples total**. There are no per-sweep fitted offsets or gains.
Unfitted onsets and the original full responses remain in the source/input bundles.

## Fit and visual review

The [joint traces](joint-state-fit.png) and [all residuals](all-state-residuals.png)
were opened and inspected. The return generally falls over the observed tens and
hundreds of milliseconds, which the state candidate follows. The second-pulse
residuals remain structured: the 1000 ms gap is overpredicted over much of its
pulse, while the 4000 ms gap is underpredicted early. Short-gap return currents
initially go below baseline, which the candidate misses.

The first-pulse discrepancy is especially informative because it precedes each
recovery gap. The first response at phase 60 ms is 35.999 pA after baseline
centering in sweep 123 and 61.165 pA in sweep 141. The shared prediction is
40.159 pA in both. Increasing gap and acquisition order are confounded in this
recording; this pattern cannot be attributed to a future recovery command.
Prior history, initial state or recording changes remain relevant. Fitting only
second-minus-first current did not expose this absolute-response requirement.

| Quantity | Joint fit |
| --- | ---: |
| Fully available amplitudes at pulse phase zero (pA) | 160.09, 508.39 |
| Pulse decay constants d (ms) | 24.78, 693.45 |
| Gap recovery constants r (ms) | 88.38, 503.91 |
| Shared initial fractions f | 0.04336, 0.00701 |
| Shared pulse term C (pA) | 36.27 |
| Return current factor rho | 0.2570 |
| Holding factors eta | approximately 0, 0.5790 |
| Holding relaxation constants s (ms) | 196.21, 1148.79 |
| Shared off time constant (ms) | 91.44 |
| First-pulse residual RMS (pA) | 7.258 |
| Second-pulse residual RMS (pA) | 14.295 |
| Return residual RMS (pA) | 3.846 |

The first eta is at its zero bound, which is admissible in this candidate but
does not establish a physical channel assignment. The scaled Jacobian has
numerical rank 15 at relative tolerance 1e-8; singular values are retained.
This local sensitivity calculation is not proof of biological identifiability
or confidence intervals from independent samples. The densely sampled currents
are not independent experimental replicates.

The slow gap recovery parameter remains near 0.5 seconds compared with the
earlier difference-only fit. The fast gap parameter changes from about 60 to
88 ms when the absolute responses and return constrain the candidate. Neither
comparison establishes a qualified rate. Phase-zero amplitudes here must not
be compared directly with the earlier fit's phase-10 amplitudes.

## Holding-return excursion

[The direct excursion view](holding-excursion.png) preserves sweep 141's current
and command together. At post phases 130.32 and 141.68 ms, baseline-centered
current reaches -121.335 and +120.540 pA. The amplifier command remains near
-19.994 mV. No cause is assigned and no samples were discarded or smoothed.
The fit includes this excursion. The overview's return scale was expanded to
show its complete range and reopened; the residual heatmap also retains it.

## Verification and scope of the next protocol

The [terminal run receipt](run-receipt.json) records normal completion in 29.89 s,
below the 300 s wall cap; the optimizer used 45 evaluations and reported success.
The fit itself took 28.11 s. Execution used local CPU closed-form array algebra,
with no neuronal rollout, remote job or animal-derived parameter source.

The affected suite passes **152 tests**. The new state helper has 100% line
coverage. Tests compare a separately calculated population-wise state oracle,
zero/long-gap limits, pulse and holding endpoints, long-hold return to baseline,
distinct pulse/return clocks, malformed/incomplete observations and a synthetic
joint-current fit. Stored predictions, state endpoints and source sample
membership are independently verified in the analysis receipt.

Only the commands and notebook holding metadata for calibration tail sweeps
99-103 were additionally inspected. They step to about +70 mV, then return to
commands from -90 to -150 mV. Their response arrays remain unopened. Those
commands lie outside this candidate's current coverage, so no direct prediction
or validation claim was manufactured by substituting +60/-20 mV dynamics.
The command-only inventory is [retained here](next-protocol-commands.json).

Next, separate estimation of the initial state from prediction of later currents,
using the observed first response to test whether state variation explains the
remaining mismatch. Any such reanalysis is still calibration, because these
records already contributed to the global fit. Broaden the voltage constraints
with the remaining calibration protocols before deploying a voltage-dependent
mechanism. The original whole-cell split is unchanged and the reserved external
donor's currents remain unopened. Whole-cell physiology, anatomy transfer and
the complete 104-cell driven window are still required for the full objective.
