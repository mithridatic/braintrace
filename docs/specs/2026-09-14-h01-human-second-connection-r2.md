# Second human connection: observed spike input and absent-spike window

Continuation of the approved human-only qualification plan. The original
six-command experiment stopped before any amplitude fitting. Keep its driver,
stderr and terminal receipt unchanged in `human-second-connection`.

## Observed protocol discrepancy

In calibration sweeps 0-14 of 0077, IN3 has four train spikes and one recovery
spike. There is no spike at the fifth putative train pulse. Its epoch has duration
40100 samples, period 10000 and width 150. The current modulo reconstruction
inserts a final truncated 100-sample pulse; this rule is not verified against
Clampex. The first recording had duration 40150 and its complete fifth pulse
coincided with a recorded spike. Do not infer delivered current from voltage.

Fail closed when an epoch would require a truncated pulse. Add a failing boundary
regression before changing the adapter. Keep complete-pulse behavior unchanged.
Do not implement an unverified omission rule. Primary manual and archived SDK
lookup did not resolve truncation; stop that lookup for this experiment.

## Prospective corrected experiment

Use the original five recorded upward zero crossings as synaptic inputs. Require
exactly five in every sweep, each inside the unambiguous full-width intervals
[28125,28275), [38125,38275), [48125,48275), [58125,58275),
[93225,93375). Fail on any missing or extra crossing, including validation.
Do not use inferred DAC3 current as synaptic input or invent a sixth spike.

Keep SIX observation slots per sweep, all original samples. The first four and
last slots are aligned to their recorded spikes. Slot 4 (zero based) is anchored
at source epoch index 68125, explicitly an epoch-clock observation with no
recorded spike. Keep -10 to +50 ms there, including pre-event baseline [-10,-2)
and scored 0-50 ms. This preserves the interval whose response contradicts the
original reconstructed command. Sum the model's tails from all five real spikes
in every slot; do not insert a new input for the absent spike.

Keep the same frozen cell and synaptic waveform, calibration sweeps 0-14,
validation 15-29, one linear positive strength, bounds and original >=25% RMSE
improvement / no sweep >5% worse gates. Preserve all 180 slot errors. Additionally
require the same synaptic gate on the 150 actual-spike windows alone, to prevent
the added absent-spike window from hiding a failed response prediction. Report
the absent-spike windows separately, without a claim of measured zero current.

This correction is registered before any strength fit or validation EPSP score.
All original data remain available; the failed six-command registration is not
retroactively passed. Electrical transfer remains separately evaluable and uses
the original unchanged Step command and frozen parameters. Save that result
before synaptic extraction so a later failure cannot erase it.

Use a new `human-second-connection-r2` directory, cap analysis at 120 s, inspect
raw and centered voltage for sweeps 0,1,14,15,16,29 and all slot residuals, and
independently verify source samples, clocks, linear optimum and both gates.
No population score changes from this experiment alone.

The prior mistake was treating a partial final period and partial pulse as the
same boundary case. Cover exact-width, shorter-than-width, shorter-than-period,
and full-period endings in sibling tests; reject ambiguous partial pulses.
