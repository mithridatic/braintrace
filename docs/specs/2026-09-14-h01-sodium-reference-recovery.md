# Human sodium recovery at the measured reference temperature

Approved parent: all-six-term, all-104 human-only qualification. The previous
checkpoint recovered author QC from the original release. Use it to test the
published sodium model at its 25 C reference temperature before another donor
replacement. No kinetic parameters are fitted in this experiment.

Use the nineteen human Figure 3 records and their filename-joined original QC.
Require exclude=0, RsComp>50, recovery fit quality >0.90 and voltage <=-75 mV,
as in the source plotting code. Keep every source observation with its individual
eligibility decision. Restrict the model comparison to -90 through -75 mV, the
overlap with the published model assay's voltage domain; report any unavailable
comparison separately. Do not treat these previously seen records as a holdout.

Pin the published assay model 318629-na_human.mod and whole-cell model 318563.
Their m/h rate constants agree; the whole-cell model includes a conductance ratio
that cancels in the normalized recovery comparison. Use the production sodium
rates at 25 C, no voltage shift or thermal factor. This experiment does not use
the failed 34 C thermal-scaling candidate.

Reproduce the source model assay's command durations: equilibrium at -120 mV,
2 ms at -120, 10 ms at 0, recovery at each comparison voltage for delays
1,4,9,...,144 ms, then 2 ms at 0. Evolve m/h exactly at each constant voltage
using vectorized closed-form exponentials; there is no repeated Python time-step
loop. Compute m^3*h*(141-V), with no claim of measured ionic-current amplitude.
Use all second-pulse samples at 0.02,...,1.98 ms for the inward peak, then fit
the same free-amplitude/free-offset exponential and R-squared>0.95 rule used by
the source get_tau_act. Report its fitted current-recovery constant separately
from the internal h-gate time constant. Failed fits remain unavailable.

Repeat peak extraction on 0.01,...,1.99 ms; require relative fitted-time-constant
difference <=1% before interpreting the numerical comparison. Do not alter the
threshold after results. The ideal held-voltage calculation does not reproduce
NEURON VClamp feedback, its lookup-table interpolation or the experimental clamp
tracking. Keep that boundary explicit; do not infer source-current validation.

Retain every selected human datum, model state and second-pulse waveform, all
delay peaks and per-record signed errors. Inspect plots showing individual
recordings across voltage and all model recovery cycles. Report numerical and
human-response discrepancies separately. No six-term score or donor promotion
follows from this source recovery comparison alone. Cap the local CPU analysis
at 120 s. Test analytic constant-voltage states, finite bounded gates, incomplete
peak windows and a known exponential recovery oracle in sibling tests.
