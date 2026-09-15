# A joint voltage-dependent human-current model was implemented and rejected

The new nineteen-parameter candidate was actually fitted to eleven human PAX6
calibration protocols. The time-quadrature fitting RMSE fell from 29.883933 to
9.983079 pA, but the optimizer exhausted its 80-evaluation limit and three kinetic
parameters reached the registered bounds. The result is **rejected before new
response access**. None of the six population scores changed and no donor model
was promoted. The six reserved sustained-current responses remain unopened.

The [model implementation](../../../h01_pax6_voltage_model.py) carries activation,
two availability states, two recording-transient states and a filtered observation
through complete voltage-command histories. It uses exact constant-command
solutions and vectorized affine composition, including the equal-rate filter
limit. There is no Python timestep loop. These are conditional amplifier-current
predictions; model components are not isolated ionic currents, measured patch
voltages or validated channel identities. All fitted biological inputs in this
attempt are from this human source; human-only physiological qualification remains
unproven.

## Executed fit and failed requirements

The [registration](../../../../specs/2026-09-14-h01-pax6-voltage-model.md) precedes
implementation and response prediction. Training sweeps are 70,72,74,76,78,
79,83,88,99,101,103 from specimen 840043506, session 840043481. The original source
SHA256 is `30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944`.
The [input receipt](training-inputs.json) binds all 890000 original current
samples and the command/clock arrays in [training-source.npz](training-source.npz).

The fitting objective uses 46255 original samples with registered time-quadrature
weights, representing the 880375 original scoring intervals after 35 ms. This is
not full-sample least squares. The completed candidate was separately evaluated
on **all 880375 original scoring samples**, yielding RMSE **9.995488 pA**. All
observations, predictions, residuals and conditional components are retained in
[training-predictions.npz](training-predictions.npz). Per-sweep RMS errors range
from 1.357 to 14.727 pA; no failed waveform was removed.

The [frozen candidate](frozen-candidate.json) retains all parameters and bounds:
d1=1.00000016 ms at its lower limit, r1=2999.99999999 ms at its upper limit,
and r2=10.00000000 ms at its lower limit. E=-119.999246 mV is also close to its
-120 mV bound, though outside the registered boundary-proximity threshold. These
are unsuccessful conditional estimates, not measured human constants. The
optimizer reports that the maximum function evaluations were exceeded. Normal
process termination does not override that failure.

The [decision](decision.json) enforces both failed requirements. No evaluation of
sustained sweeps 105,106,107,109,110,111 was performed. The separately reserved
external PAX6 donor and original whole-cell holdouts also remain unopened. Do not
extend the evaluation cap or widen kinetic bounds merely to get a nominal pass.

## Direct observations and the next correction

Opened [all training responses](all-training.png), [all residuals](all-training-residuals.png),
and [waveform detail](training-detail.png). The model follows much of the broad
depolarized-current decline, but misses the sharp command-transition peaks.
Sweep 74's depolarized residual changes from underprediction to overprediction;
sweep 78 retains structured differences during the decline and a persistent late
tail offset. Sweep 103's tail rises toward its late current faster than the model
initially does. These differences remain visible separately from the RMS scores.

The [control-onset comparison](control-onset-detail.png) was then opened. It
reveals a response-timing mismatch that the initial first-order observation model
cannot reproduce well. Across all eleven control onsets, measured current at
0.12 ms remains between -2.83 and +1.55 pA relative to baseline; at 0.16 ms it
rises to 12.00-18.70 pA. The model already predicts 18.7155 pA at **0.04 ms**.
Its control peak is too broad and low. The same early-response shape mismatch
appears at control turn-off. The [exact samples](control-onset-observations.json)
retain original on/off indices, pre-transition levels, all 28 phase samples and
both waveforms for every sweep. This is a post-fit diagnostic, not a registered
intervention result or an estimate of the instrument's physical delay.

The immediate next model correction should address this measured control-response
timing and shape before another kinetic fit. A delayed or richer observation
response is a candidate to test, not an established hardware explanation. The
initial mistake was assuming an immediate first-order observation response while
jointly asking the optimizer to identify biological kinetics. Prevent recurrence
by checking a proposed observation response against the control onset and offset
before fitting active-current parameters. Paired-pulse and conditioning protocols
will also be needed to constrain recovery; this fit's bound values cannot establish
those dynamics. Do not reopen the earlier initial-state-only repair or assign
the recording-transient components to membrane channels.

## Verification and runtime

The helper passes [36 sibling tests](tests.xml), with [98% line coverage](coverage.json).
Independent oracles cover leak/filter response, capacitive steps, integration of
active current, equal rates, long holds, subdivision invariance, batched state
reset and complex-step derivatives. Failure cases cover invalid parameters,
discontinuous commands, padding and missing/nonfinite queries. Future observation
model tests must also reproduce the delayed sharp control response; this helper's
mathematical checks do not assert that its observation family fits that response.

[Independent artifact verification](verification.json) checks all original training
current samples, source clocks, fitting indices/weights, every scoring observation,
all per-sweep errors, 1100 recomputed predictions, source/candidate hashes and the
failed access gate. The [terminal receipt](terminal.json) records **378.538 seconds**
on local CPU under the 600-second cap, exit code zero and no timeout. The optimizer
failure is separately recorded. No membrane or population simulation ran.
