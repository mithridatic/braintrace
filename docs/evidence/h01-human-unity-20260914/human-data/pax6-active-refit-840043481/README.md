# Active-current refit: executed and rejected

The registered 13-parameter active-current candidate was implemented, tested and
fitted to 17 acquired human PAX6 sweeps with the accepted recording response held
fixed. Optimization converged after 32 evaluations, in 193.75 seconds of local CPU
process time (600-second cap). The fitting objective fell from 32.76223 to 10.56193
pA RMS. This is a fit to previously exposed data, not an independent prediction.

Full-sample scoring and rendering completed in 30.57 seconds (120-second cap),
covering 1,487,625 original samples after the registered baseline boundary. Its
RMS error is 10.55160 pA. All six early recording controls and the long control 83
remain within the allowed 5 percent of the frozen observer, with slight improvements.
However, the fitted reversal parameter E reaches its -120 mV lower bound. The
[registered decision](decision.json) therefore rejects the candidate before any
reserved response access. Convergence does not override this failed prerequisite.

No sustained prediction, external donor or whole-cell holdout was opened. No
physiology was qualified and none of the six population scores changed. No
parameter bounds or acceptance rules were relaxed. This refit used additional
conditioning/recovery responses, so its pooled error is not directly comparable
with the older eleven-sweep candidate's pooled error.

## Direct observation review

All five rendered figures were opened with the image viewer. They retain the
original command and current samples; DAC commands are not measured patch voltage,
and the observations are total amplifier currents, not isolated channel currents.

- [Depolarization](depolarization.png): sweep 74 is underpredicted early in its
  sustained response and overpredicted later. Sweep 78 shows the opposite broad
  residual progression. These structured errors remain after correcting the fast
  recorded response; one pooled RMS value does not describe their timing.
- [Conditioning](conditioning.png): sweep 93 again shows an early deficit followed
  by a late excess. Sweep 97 retains a positive late residual. Large edge residuals
  also remain; passing the small-control observation gate does not establish
  accurate observation of every large voltage step.
- [Recovery](recovery.png) and [onset detail](recovery-onsets.png): the common first
  pulse overpredicts the early post-transient current in 123 and 132. In the second
  pulse of 132, the human response has a shoulder above the candidate before the
  traces approach one another. In 141, the candidate rises below the human response
  after the narrow second-pulse transient. The long -90 mV interval retains a
  baseline discrepancy. The large later human excursion remains in the original
  arrays; it was not deleted or replaced by the model.
- [Tails](tails.png): the candidate follows the broad depolarizing response but
  retains sustained and return-edge residuals. The initial source tails remain
  negative across the observed commands. The bound-hitting fitted E is not an
  independently measured reversal potential.

The remaining uncertainty includes active-state voltage dependence, the holding
initial-state assumption, and observation/patch behavior during large commands.
This result does not isolate a unique missing molecular mechanism. The present
candidate ties the voltage dependence of relaxation times to the availability
asymptote; separate relaxation voltage dependence is a testable follow-up hypothesis,
not a correction established by this fit. Do not repeat this unchanged fit or
widen E merely to obtain an interior solution.

## Implementation and verification

The helper expands exact constant-command gated current into exponentials and
propagates the fixed four-pole Bessel modes across every command segment. There is
no Python simulation timestep loop. Source export preserves all original samples;
selected samples and following-interval fitting weights are saved separately.
Every real optimizer trial is in progress.jsonl, including parameters and error.

134 affected tests passed. The new active-current helper has 100 percent measured
coverage; the frozen recording helper has 98 percent. The exporter regressions
also passed, but this invocation did not collect exporter coverage because that
test module imports it under a different module name. No exporter code was changed.

Edge cases tested include zero conductance, independent convolution quadrature,
filter/gate carry and command subdivision, independent batch starts, very long
holds, trailing padding, original-time coverage, all thirteen complex derivatives,
malformed inputs and nonfinite output. These numerical checks do not establish
human physiology. The frozen observer predictions and original source clocks were
matched during full-sample review; source, specification and implementation hashes
are bound in frozen-candidate.json and verification.json.

The prior mistake was treating an insufficient observation model as part of a
joint active-kinetic fit. This attempt fixes the observer before fitting and retains
its separate acceptance gate. The remaining active-model failure is preserved,
rather than counted as a biological improvement or hidden by a smaller fitting error.
