# Matched human PAX6 conditioning protocols

All nine total/prepulse test-voltage pairs are now prepared from human specimen
840043506, session 840043481. Their direct differences do not qualify a
voltage-dependent decay law. A same-command comparison also finds different
responses across protocol families, so a stationary starting state cannot simply
be assumed. This checkpoint changes the next inference step; no H01 score changes.

## Correction to the preceding review

The earlier [current-preparation review](../pax6-840043481/README.md) incorrectly
described prepulse sweep 89 as testing near -90 mV while total sweep 70 tested
near -50 mV. Both test near -50 mV over **1100-2100 ms**. The prepulse conditions
near -20 mV over 1000-1100 ms; the return to -90 mV starts at 2100 ms.
The earlier sealed report is preserved as historical evidence. Its pairing
statement is superseded by the stored commands and this explicit correction.

The mistake was a misidentified command interval. The new helper checks every
sample in the named test window and matches unique partners by measured command,
then checks holding, conditioning, return and original clocks. Tests include
conditioning above the test voltage, as occurs for the -50 mV pair. A family
index or a visually selected return transition no longer determines the pairing.

## Sources and direct observations

NWB SHA256:
`30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944`.
All 18 full exports and nine paired bundles retain original 25 kHz samples.
The source JSON binds the NPZ and the pair JSON binds both source exports.
No filter, interpolation, liquid-junction or series-resistance correction was
applied. Command voltage is not measured patch voltage. Differences include
recording/history effects and are not established isolated potassium currents.

The [commands](commands.png), [original currents](paired-currents.png) and
[differences](differences.png) were opened and inspected. Commands match exactly
over the test window at nine levels from -49.987 to +70.013 mV. At -19.987 mV the
conditioning plateau continues into the test window without another voltage step.
The onset artifacts are retained in the paired and full-source arrays.

Near -50 and -35 mV, differences beyond the onset are small compared with baseline
shifts. Near -20 mV, total-minus-conditioned current stays negative. Near -5 mV,
it starts negative and fluctuates toward zero. High positive commands show a
larger positive early difference, followed by smaller, fluctuating differences.
At +70 mV the difference declines, then grows again through parts of the later
window. That later behavior cannot be represented exactly by positive decays.

Initial total-minus-conditioned baseline differences range from -6.514 to -1.034
pA. Subtracting those baseline differences is retained as a sensitivity case,
not an established correction for drift or a bound on true ionic current.

## Same-command and acquisition-history check

[The history figure](same-command-history.png) was also opened and inspected.
All nine prepulse sweeps share the same -90 to -20 mV conditioning command.
Total sweep 72 supplies another such transition. The figure aligns the actual
command onsets, preserving both source clocks and a separate full-transient view.

At 20 ms after the actual step, sweep 72 measures 21.250 pA, while the nine
conditioning responses range from 43.125 to 60.000 pA. After subtracting each
sweep's own initial baseline, the corresponding values are 38.734 pA and
54.809-72.799 pA. The post-transient total response lies below the later group
through the displayed early window. The longer comparison of sweeps 72 and 91
retains a difference after onset alignment and baseline centering.

This does not uniquely diagnose slow activation, recovery, rundown or clamp error.
The holding interval since the small early test pulse differs by 100 ms, prior
stimulus histories differ, and the acquisitions occur about 81 seconds apart.
The notebook's test-pulse steady-state resistance changes across sweeps; its
series-resistance field is zero and compensation is disabled, so zero must not
be treated as measured zero access resistance. The instrument fields are retained
as observations. A stationary initial state and unchanged recording transfer
function have not been established.

[History observations](history-observations.json) also bind all 19 recovery
acquisition clocks. Consecutive stored traces have unobserved gaps of about
0.500-2.202 seconds. Commands and currents during those gaps are absent from the
exports. A continuous-history model would need an explicit assumption there;
these are not measured holding trajectories.

## Conditional fit result and interpretation

Each of the nine pair differences was fitted in both baseline scenarios using
all 24,250 samples over phases 10-980 ms. The model has two nonnegative
exponentials and no constant offset. All 18 fits, their parameters, predictions,
optimizer status and residuals are retained. Total optimizer time was about
1.39 seconds on local CPU; there was no neuronal simulation or remote compute.

[Fit review](fit-review.png) and [residuals](fit-residuals.png) were visually
inspected. At the four lowest commands the unadjusted positive-amplitude model
collapses to zero; the negative observations cannot be explained by that model.
Baseline centering does not resolve the -20/-5 mV mismatches. Higher-command fits
capture some early decline but retain structured residuals and slow-bound limits.

| Test command (mV, rounded) | Unadjusted decay constants (ms) | Interpretation |
| --- | --- | --- |
| -50, -35, -20, -5 | At lower bounds with zero amplitudes | No identified decay constants |
| +10 | 37.6, 10000 | Slow component reaches upper bound |
| +25 | 2.1, 738.8 | Fast amplitude only 3.4 pA; weak component, not a qualified fast rate |
| +40 | 46.3, 1375.7 | Interior solution; residual structure and source-state ambiguity remain |
| +55 | 118.3, 10000 | Slow component reaches upper bound |
| +70 | 37.5, 10000 | Slow component reaches upper bound; later rise remains unexplained |

The 10000 ms entries are fitting limits, not measured time constants. Optimizer
active-bound flags are reported literally. The history report separately records
distance to bounds, including the -50 mV sensitivity fit that approaches 10000 ms
without receiving an active-bound flag. Optimizer convergence does not imply
identifiability. These fits are not promoted into a channel implementation.

## Verification and next action

The full affected suite passes: **119 tests**. The new conditioning helper has
100% line coverage. Its 21 tests cover analytic differences and decay curves,
baseline retention, source timestamps, command matching, unequal clocks,
extra transitions, incomplete/off-grid windows, invalid arrays and ambiguous
partners. Existing source binding and external-access rejection tests also pass.

Before fitting a shared voltage-dependent mechanism, compare the test-pulse and
leak-control observations across the protocol sequence and explicitly constrain
initial states. The same-command discrepancy must be explained or represented
as unresolved source variability; a single fit with identical initial states
cannot be justified by command matching alone. Then evaluate a candidate jointly
against total, conditioning, sustained and recovery responses.

The reserved external specimen's currents remain unopened. The original
whole-cell calibration/holdout split is unchanged. Whole-cell physiological
validation, audited anatomy transfer and the 104-cell driven window still remain.
