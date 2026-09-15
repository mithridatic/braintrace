# Active-current refit with a frozen recorded observation response

Continues the approved human-only donor plan. The preceding turn made progress:
the recorded-filter correction passed its specified control-prediction gate.
The original six-term, 104-cell acceptance contract remains unchanged.

Freeze the observation candidate from e4ec8dcd, including its five parameters and
fixed four-pole 2000 Hz Bessel cutoff. Do not fit the observation response again.
Retain the previous nineteen-parameter candidate and its failed decision.

## Model

Use the same thirteen active-current parameters, initial values and bounds as
the first thirteen entries in h01_pax6_voltage_model.py. Activation m has a
logistic voltage asymptote and constant time constant. Two availability states
have a shared logistic asymptote and distinct voltage-dependent relaxation
constants. Current is (V-E)*m^2*(g1*h1+g2*h2+gs), relative to initial holding
current. Initial gates equal the first holding-command asymptotes. This remains
a conditional mathematical candidate, not identified molecular channels.

Integrate gate states exactly over every constant-command segment. Expand active
current into exponentials and propagate their effects through the actual Bessel
pole pairs, including filter state between command segments. Add the already
frozen passive/recording response. Apply its fixed observation latency to the
entire recorded prediction. Commands are not measured patch voltages; recording
components cannot be deployed as membrane channels. Use vectorized analytic
composition, with precomputed command/observation geometry and no Python model
timestep loop. Preserve all original times and currents.

## Data and fit

Use the pinned human PAX6 specimen 840043506/session 840043481, SHA256
30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944.
Training: previous total sweeps 70,72,74,76,78, controls 79,83,88, tails 99,101,103;
add already acquired conditioning sweeps 89,93,97 and recovery sweeps 123,132,141.
Use actual command segments, not family positions or assumed voltages. No new
held-out response is opened for this fit. The three selected recovery conditions
do not supersede the prior all-nineteen recovery observations or their failures.

Center each complete response on its original 35-44 ms control baseline. Fit
every 25th original sample after 35 ms, adding every original sample during the
first 2 ms after each command transition. Use square-root following-interval
counts as time-quadrature weights. The fixed recording response no longer requires
estimating its parameters from densely sampled long onset windows. This is a
registered numerical fitting objective, not full-sample least squares; score the
final candidate on every original sample after 35 ms and retain original arrays.

One starting point (the original active initial values), at most 80 least-squares
evaluations, complex-step derivatives and a 600-second local CPU fit-process cap.
Append every real parameter trial and objective to a progress JSONL so a timeout
retains actual attempted parameters. Do not restart an unchanged timed-out run.
Full-sample evaluation/rendering is a separate bounded local step, capped at
120 seconds; no population or membrane rollout. Preserve nonconvergence and
boundary failures even if the response error improves.

## Access and acceptance

Before opening sustained prediction sweeps 105,106,107,109,110,111, require
optimizer convergence, finite full-sample predictions, no active-current
parameter within 1e-6 normalized distance of its bounds, and preservation of
the accepted recording correction: for each of its six excluded early controls
and long control 83, no more than 5% worsening versus its frozen prediction.
The three training conductance amplitudes must also be interior; zero amplitude
would leave associated kinetics unobserved. A failed prerequisite keeps all new
prediction responses sealed. Do not adjust bounds or access rules after seeing
the result.

If these prerequisites pass, freeze candidate, command-only protocol metadata
and full predictions before opening the six sustained responses. Retain the prior
prediction gate: at least 25% pooled full-sample RMS improvement versus this
candidate's frozen passive/observation-only prediction and no individual sweep
over 5% worse. This same-specimen test is not external donor validation. Passing
permits further kinetic qualification only; the original whole-cell holdouts,
external donor and six population scores remain unchanged until their own gates
are met. Remaining recovery observations must still be predicted before deployment.

## Verification

New helper and sibling tests must independently check passive-limit equivalence
with the accepted correction, active-current convolution against quadrature,
filter-state carry, subdivision invariance, long holds, batch resets, complex-step
derivatives for every fitted parameter, missing coverage and invalid inputs.
Target >90% helper coverage. Bind frozen observation/helper hashes, raw source
membership, fitting masks/weights and full-sample results. Inspect complete
responses and residuals, with onset and paired-pulse detail, before interpreting
the candidate. Do not substitute a pooled error for those direct observations.
