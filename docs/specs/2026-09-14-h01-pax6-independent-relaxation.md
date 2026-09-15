# Independent relaxation voltage dependence: prospective candidate test

Continues the approved all-six/all-104 human-only plan. The previous turn was
progress: commit 3976d700 contains an executed, converged active fit and a preserved
rejection. It produced no qualified physiology or population score gain.

## Discriminating change

The previous model ties availability h_inf(V) and relaxation time to one voltage
curve: tau_j = d_j + (r_j-d_j)*h_inf(V). After the observation correction, its
depolarizing and conditioning responses still show opposite early/late residuals,
and recovery responses retain a shoulder that it misses. Its reversal E reaches
the lower bound. These observations motivate testing that coupling; they do not
prove a unique channel mechanism or exclude optimization uncertainty.

Introduce a separate logistic q(V) = 1/(1+exp((V-vtau)/ktau)) only in
tau_j = d_j + (r_j-d_j)*q(V). Keep the activation, shared availability asymptote,
initial holding equilibrium, current expression and all existing thirteen
parameters unchanged. Two new parameters vtau, ktau start at -20 mV, 10 mV and
have bounds [-60,20] mV and [2,35] mV. The old candidate is an exact special case
when vtau=vh and ktau=kh. Check that reduction over complete multi-pulse responses.
The new logistic is a conditional mathematical family, not animal-derived kinetics
or a claim of identified human channels.

Keep the accepted five-parameter observer and fixed four-pole 2000 Hz Bessel
response frozen. Keep E within the unchanged [-120,-50] mV interval. Preserve the
previous implementation and failed evidence without modifying their bytes.

## Data, numerical budget and qualification

Reuse exactly the 17 acquired sweeps and preserved original arrays, baseline
indices, fit indices and time-quadrature weights from pax6-active-refit-840043481.
No response access is needed for fitting. Preserve source/observer/candidate and
dependency hashes. Use original thirteen initial values plus the two registered
new initial values, one starting point, complex-step least-squares derivatives,
at most 80 evaluations and a 600-second local CPU fit-process cap. Record every
real trial. The process cap includes preparation. A separate full-sample scoring
and plotting process is capped at 120 seconds. Do not restart unchanged failures.

All earlier access prerequisites remain: optimizer convergence, finite complete
predictions, every fitted parameter more than 1e-6 normalized distance from its
bounds, and no greater than 5 percent worsening versus the frozen observer on each
of six early controls and long control 83. Report the prior/new RMS comparison
using exactly the same 1,487,625 samples, and inspect the original traces and
residuals rather than interpreting a pooled improvement as physiological evidence.

If prerequisites pass, freeze the candidate and complete command-only predictions
for sustained sweeps 105,106,107,109,110,111 before opening their responses. The
same-specimen gate remains >=25 percent pooled full-sample RMS improvement versus
the frozen passive observer and no individual response more than 5 percent worse.
External donor and whole-cell holdouts remain closed. A failure leaves prospective
responses unopened; a pass only permits further kinetic qualification, including
remaining recovery observations. Neither outcome alone changes a population score.
If eligible, the prospective prediction freeze, response scoring and rendering
use one further local CPU process capped at 120 seconds. The three possible
process caps together are fourteen minutes; no external compute is purchased.

## Verification and direct observations

Use exact constant-command algebra with vectorized affine/filter propagation, no
Python simulation timestep loop. Tests must cover reduction to the archived model,
passive equivalence, all fifteen derivatives, the intended separation of asymptote
and relaxation, independent numerical convolution under the new voltage curve,
invalid parameters, and >90 percent coverage. Reuse the already tested command
validation, long-hold and filter-carry checks. Inspect full depolarization,
conditioning, tail, recovery and onset plots on their original clocks. Retain
the seven observer checks and the original rejected fit for comparison.
