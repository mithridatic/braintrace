# Human EPSP prediction with the cell response held fixed

Approved parent: all-104 human-only qualification, including synaptic parameters
and recorded human responses. Test a current-based synaptic candidate against
the acquired human paired recording before considering any deployment.

## Inputs and split

Use only ABF 359652 and the frozen electrical-response candidate in
human-paired-passive/frozen-candidate.json, SHA256
a4cec3ab1f9f3ccb012586f77f78cd42941b38b5ddbf7330064389721e4d8c60.
Keep R=72.55903367837949 MOhm and membrane response tau=13.077357533655515 ms
fixed. Exclude the fitted instantaneous series term: synaptic current enters
the membrane and is not injected through the electrode's series path.

Use sweeps 0-14 for fitting and 15-29 for validation, preserving the electrical
experiment's split. Earlier plots included sweeps 0, 1 and 29, so this is a
within-recording prospective comparison, not blind or independent-donor evidence.
Freeze the new candidate before extracting/scoring validation EPSPs.

Detect every presynaptic upward zero-mV crossing from original IN0 samples.
Require six complete events per sweep, consistent with the observed source
protocol; stop on an unexpected count or incomplete window rather than dropping
events. Retain original IN2 voltages and IN0 voltages from -10 to +50 ms around
each event, source indices and both original clocks. Each event's fixed baseline
is the mean over source indices [-500,-100), corresponding to -10 to just before
-2 ms. Score all samples from 0 through +50 ms inclusive. Keep all spontaneous
deflections and pre-existing responses; these contribute residuals and are not
silently attributed to the evoked input or removed as outliers.

## Candidate

The effective synaptic current is a positive normalized difference of exponentials:
I(s) = A * (exp(-s/td)-exp(-s/tr))/N for s >= 0 and zero otherwise,
where s is time after the recorded presynaptic crossing plus fitted delay,
td=tr+gap, and N makes the current peak equal A. Filter this current by the
frozen membrane impulse response R/tau * exp(-s/tau)/1000, using the exact
closed-form convolution. Sum contributions from all six actual spike times.
Subtract the candidate's own mean over each pre-event baseline window, matching
the human observation operator and retaining tails from earlier events.

Use one common A, tr, gap and delay for all events and all sweeps. No separate
event gains, release states, per-sweep kinetic parameters or response-fitted
offsets. This tests a fixed-strength synapse; it does not assume that biological
release is deterministic. Unmodeled spontaneous inputs and release variability
remain visible in individual residuals. The fitted current is an inferred
effective input under the frozen cell model, not a measured ionic current,
identified receptor or voltage-dependent conductance.

Fit log A, log tr, log gap and linear delay by ordinary least squares over all
calibration events and samples with equal sweep/event weight. Initial values:
A=60 pA, tr=0.3 ms, gap=2.7 ms, delay=1 ms. Bounds: A [0.01,2000] pA,
tr [0.05,5] ms, gap [0.05,95] ms, delay [0,5] ms. These are numerical hypothesis
bounds, not sourced physiological constants. One start, at most 60 optimizer
evaluations, local CPU and a 120-second wall cap. All time evolution is evaluated
by closed-form vectorized convolution, with no Python simulation-step loop.

## Decision and evidence

Reference prediction is the same per-event baseline with no evoked response.
Advance for further human qualification only if optimization converges without
active bounds, validation combined RMSE improves by at least 25%, and no
individual validation sweep worsens by more than 5%. Report all 180 event errors,
all 30 sweep errors and per-ordinal errors; no observation is discarded. A pass
does not qualify another donor, voltage range, geometry, receptor mechanism or
any six-term population score.

Preserve original samples, all candidate evaluations, frozen parameters, inferred
current and voltage predictions. Plot first/second/last events in sweeps 0, 1,
14, 15, 16 and 29, plus every residual with common scales. Inspect spontaneous
overlap, train dependence, onset and decay separately from aggregate scores.
Tests must cover analytic convolution against independent numerical quadrature,
the coincident-time-constant limit, current normalization/units, causality,
linear superposition, earlier-event tails, baseline subtraction and invalid
inputs. Independently verify source sample membership and the frozen decision.
Do not tune after validation or relax a failed criterion.

## Correction after the pre-fit integrity stop

The initial run stopped in calibration sweep 2 before optimization or validation.
Its executed specification and terminal record remain under human-synaptic-prediction.
There are six commanded spikes, but a small falling-phase recrossing adds a
seventh upward zero crossing. The raw samples show -0.2136, +0.3052, +0.0610,
then -0.4578 mV at indices 68279-68282; this is not a second full spike.

The command overlay also exposed a pyabf 2.3.8 reconstruction defect: its pulse
count floors duration/period and omits a complete final pulse when the epoch ends
before the final full inter-pulse period. The source epoch is 40,150 samples,
period 10,000, width 150: it contains five pulses, not four. Add a narrow local
Step/Pulse epoch adapter using source indices modulo period, rejecting unsupported
epoch types. Reproduce the missing pulse in a failing test before correcting it.
Do not alter the installed dependency or the already sealed evidence.

For the new run prefix human-synaptic-prediction-r2, associate spikes with the
six reconstructed positive command pulses, requiring exactly one upward zero
crossing within each pulse. Retain and report all unassigned crossings and their
original samples; no voltage samples are removed from the comparison. This is a
conditional prediction of the six evoked events, not an assertion that all
unassigned activity is nonexistent. Additional crossings must be visually
examined. Preserve the source command and the prior incorrect parser command.
The fitting model, bounds, split, observation baseline and rejection gate remain
unchanged. The passive -10 pA Step epochs are unaffected by this Pulse defect.
