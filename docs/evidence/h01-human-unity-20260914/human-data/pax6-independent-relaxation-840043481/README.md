# Independent relaxation fit: improved error, failed qualification

The registered 15-parameter candidate completed in 227.90 seconds on local CPU,
with optimizer convergence. Full-sample review completed in 33.14 seconds. Every
one of the same 1,487,625 original samples, commands, baseline indices and fitting
weights was retained. Full-sample RMS error fell from 10.55160 to 8.27016 pA
(21.6217 percent). This is a training comparison with two additional parameters,
not independent physiological validation.

The [decision](decision.json) rejects the candidate before new response access.
Four parameters are within the registered 1e-6 normalized boundary limit: gs,
kh, E and vtau. All seven frozen-observer checks still pass. The previous failed
candidate, observer and source arrays are unchanged. No reserved sustained,
external-donor or whole-cell response was opened, and none of the six population
scores changed. No parameter bounds or acceptance thresholds were relaxed.

## Direct observations

All six PNGs were opened with the image viewer. The full plots preserve original
DAC commands and total amplifier currents, rather than treating either as measured
patch voltage or isolated channel current.

- [Depolarization](depolarization.png): the broad early/late mismatch in 74 is
  reduced, though a positive middle residual remains. Sweep 78's late deficit is
  smaller, while an early negative residual remains. Sweep 72 worsens from 5.96
  to 7.20 pA RMS; improvement is not uniform across voltage.
- [Conditioning](conditioning.png): 93 improves from 19.30 to 8.37 pA RMS. Its
  early deficit persists, but the previous broad late excess is largely removed.
  Sweep 97 still has structured early negative and later positive residuals.
- [Recovery](recovery.png) and [onset detail](recovery-onsets.png): the first-pulse
  excess in 123/132 remains. The second pulse in 132 still misses the observed
  shoulder. In 141, the new curve is above much of the later early response after
  the narrow transient; whole-sweep RMS worsens from 11.76 to 12.81 pA. The long
  holding discrepancy and large later human excursion are retained.
- [Tails](tails.png): errors improve for the three fitted tail sweeps, but broad
  conditioning residuals and return-edge errors remain.
- [Small holding controls](holding-controls.png): the -20 mV sweeps have a larger
  late control-pulse current than the earlier -90 mV example 103. The active model
  adds part of that response. The late control level also changes between 123,
  132 and 141, despite the same holding command. Holding voltage is confounded
  with acquisition order; this comparison does not prove a voltage-dependent
  recording error or authorize retuning the frozen observer.

## What limits the next fit

A separate post-fit local sensitivity calculation completed in 7.86 seconds under
a 120-second process cap. It did not optimize, access new responses, or change a
gate. The normalized response derivatives for availability midpoint vh and slope
kh have cosine 0.9999999818. The weakest singular direction is dominated by those
two parameters; the smallest/largest singular-value ratio is 3.16e-6 under full
bound-range parameter scaling. See [the measured sensitivity](local-sensitivity.json).

This is local evidence that the fitted recordings poorly distinguish those two
parameters at this candidate. It is not a global impossibility proof or a confidence
interval, especially at an active parameter boundary. Simply adding parameters or
widening the bounds is not justified by this result. Further work should first
identify observations that separate availability voltage dependence and acquisition
state, rather than repeating this same fit.

The shared-asymptote/time-constant coupling was a model assumption, not a measured
human relation. Separating it improved the fitted response but did not resolve
qualification. The remaining failure is preserved instead of being hidden by the
pooled improvement.

## Verification

91 affected tests passed. The new helper has 100 percent measured coverage, the
archived active helper 100 percent, and the observer 98 percent. Tests cover exact
reduction to the archived family, passive equivalence, independently integrated
convolution, all fifteen derivatives, the intended asymptote/relaxation separation,
state carry, long holds, invalid inputs and nonfinite output. Source clocks,
observations and fitting weights match the previous run exactly. Candidate,
observer, implementation and dependency hashes are retained. These are numerical
and evidence-integrity checks, not biological qualification.
