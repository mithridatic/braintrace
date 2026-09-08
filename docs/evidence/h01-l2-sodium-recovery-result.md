# Slower sodium recovery has a small first-interval effect

The factor-two intervention passes its declared prediction. The first
interval increases from 26.502158 to 26.735594 ms. The human interval is
33.851739 ms. Absolute error decreases by 0.233436 ms, but remains
7.116144 ms. First onset changes by less than 5e-11 ms and first peak
by less than 3e-11 mV. Both changes are within the declared limits.

Both runs contain five events. The second interval decreases from
158.852803 to 150.026468 ms, against the human 220.403496 ms. The third
decreases from 293.203654 to 291.167656 ms, against 307.406282 ms.
Both errors worsen. The first minimum becomes 0.177854 mV more negative.
This minimum was an observation, not an acceptance condition.

The [prediction audit](h01-l2-sodium-recovery-result.json) retains each
event. The [human residuals](h01-l2-sodium-recovery-human-residuals.json)
retain every interval and each matched onset, peak, and duration error.
The default-one source control passed exact sample equality before this
intervention. Source hashes, genome, geometry, input, and solver match
between the two intervention runs. Only the recovery factor changes.

This result supports a conditional model effect. It does not identify
a measured human recovery time or qualify the cell. The candidate passes
the [independent solver check](h01-l2-sodium-recovery-solver-comparison.md).
The candidate also passes the
[factor-three to factor-nine spatial check](h01-l2-sodium-recovery-spatial-comparison.md).
These checks apply only to this input and candidate. The next decision must account for
the small early benefit and the worsened later intervals before selecting
another parameter. The reserved human response remains outside fitting.
