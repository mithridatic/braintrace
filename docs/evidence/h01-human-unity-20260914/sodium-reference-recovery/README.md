# Reference-temperature sodium recovery: common retuning cannot close individual errors

The fixed published sodium rates reproduce the human mean recovery closely at
25 C, while individual recordings differ substantially. Across 76 eligible source
measurements the model RMSE is 4.912241 ms. The smallest possible RMSE for ANY
single shared prediction at each voltage is 4.909465 ms: even an unrestricted
shared curve could improve this error by only **0.0565%**. This is a diagnostic
bound using these observations, not independent validation or a population score.

| Recovery voltage (mV) | Model fitted current tau (ms) | Human range (ms) | Model minus human mean (ms) |
| --- | ---: | ---: | ---: |
| -90 | 6.004231 | 2.719-10.792 | +0.039102 |
| -85 | 8.349770 | 3.635-15.676 | -0.094949 |
| -80 | 12.154144 | 3.908-23.776 | +0.299788 |
| -75 | 17.556272 | 5.643-37.582 | -0.092897 |

Do not launch a global 25 C recovery-rate fit to repair these individual errors.
It cannot overcome their spread under a shared prediction. This result does not
identify the spread's biological or measurement causes. It also does not resolve
the 34 C AP-clamp discrepancy, the additional use-dependent process implicated
by the donor spike train, or the missing original channel currents.

## Registered calculation and source boundary

The [registration](../../../specs/2026-09-14-h01-sodium-reference-recovery.md)
fixes all kinetic parameters and uses the [recovered author QC](../human-channel-history/README.md).
Every one of the nineteen human records contributes four eligible points, at
-90, -85, -80 and -75 mV. All 133 source conditions are retained with eligibility
reasons in [result.json](result.json); the remaining 57 conditions are outside
the author/model comparison criteria or missing. No failed fit or recording was
silently removed and no validation split was changed.

The source model command is -120 mV for 2 ms from equilibrium, 0 mV for 10 ms,
the specified recovery voltage for 1,4,9,...,144 ms, then 0 mV for 2 ms. The
vectorized constant-voltage solution retains all 48 second-pulse waveforms and
m/h states. The peak observation window is strictly 0.02 to 1.98 ms, followed
by the source free-amplitude/free-offset exponential fit. The inward proxy is
m^3*h*(141-V); it has no fitted absolute conductance or recorded current magnitude.
The [arrays](model-observations.npz) preserve all states, proxies and delay peaks.

The fitted current-recovery constant differs from the internal h-gate constant
by at most 0.001156 ms. Thus comparing a gate tau with a fitted current tau is
not the explanation for this particular ideal-command discrepancy. Refining
the peak grid to 0.01 ms changes fitted tau by at most 0.002391%, within the
registered 1% numerical criterion. All four fits have R-squared above 0.999999998.
The source helper subtracts its final peak and shifts delay origin before fitting;
the free amplitude and offset are retained here as in that operator.

This is an ideal held-voltage calculation, not a reproduction of NEURON VClamp
feedback, source table interpolation or measured experimental clamp tracking.
Human observations are published fitted constants, not recovered raw currents.
No 34 C temperature factor, voltage shift or failed thermal candidate is used.
The assay [source model](318629-na_human.mod) and [whole-cell source variant](318563-na_human.mod)
are preserved with checksum verification. Their rates agree; the whole-cell
variant's additional conductance ratio cancels in this recovery-time comparison.

## Direct review and checks

Opened [all human recovery observations](all-human-recovery.png) and
[model recovery cycles](model-recovery-cycles.png). Most recorded curves increase
across depolarized recovery voltages, with substantial differences in slope and
level. H20.29.184.11.42.05 remains above the model and reaches 37.58 ms at -75 mV,
where the model predicts 17.56 ms. H20.29.185.11.44.08 stays well below it.
H20.29.188.11.42.06 changes direction near -80 mV before rising sharply at -75 mV;
that feature remains visible. A smooth common curve cannot reproduce all these
individual observations. The model cycles show higher h availability after longer
recovery and nearly identical m trajectories during the second pulse; longer
depolarized recovery still leaves less available current than at -90 mV.

The [shared-prediction bound](shared-prediction-bound.json) follows the exact
sum-of-squares decomposition into deviations around each voltage's human mean
plus squared model bias. It supplements, rather than replaces, the individual
observations and errors. No new biological parameter was fitted to those means.

[Independent verification](verification.json) checks all 133 source conditions,
76 selected errors, 24 rate values against constants read directly from the
published MOD file, 9,504 gate values, the numerical gate and the error bound.
The new helper passes [18 sibling tests](tests.xml) with [100% coverage](coverage.json).
Tests include analytic relaxation/composition, every assay delay, strict endpoint
coverage, an independent exponential oracle, gain invariance, invalid states and
failed/unidentifiable fits. The [terminal receipt](terminal.json) records 3.361 s
on local CPU, below the 120 s cap.

No source kinetic parameter, donor candidate, population score or physiology
qualification changed. The useful decision is to stop global 25 C recovery
retuning and retain the measured individual differences when selecting a donor
mechanism. This does not establish that arbitrary per-record fitting will predict
new human responses; that requires separately reserved observations.
