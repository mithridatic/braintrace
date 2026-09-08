# Somatic Ca_LVA conductance and the first voltage minimum

The intervention halves somatic Ca_LVA conductance in the slope-5 candidate.
Axonal conductance and all other model parameters stay fixed. Both runs
use mesh factor 9 and CVode tolerance 1e-10. The audit checks recorded
settings against the unchanged-conductance controls.

The directional test passes at both inputs. The first minimum becomes
at least 1 mV more negative, and the first two peaks remain positive.

| Input, nA | Change in minimum, mV | New minimum, mV | New time from peak, ms | Human minimum, mV | Human time from peak, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.19 | -1.845575 | -74.664190 | 2.288921 | -78.843750 | 0.760000 |
| 0.27 | -2.884558 | -72.679393 | 2.025907 | -78.906258 | 0.700000 |

The control times from peak were 2.129655 and 1.870230 ms. The new minima
are deeper but later. A later, deeper minimum does not by itself prove
a slower fall at matched voltage. Minimum depth and minimum time are
separate observations, and both still differ from the human trace.

First-event onset errors are +7.477404 and -0.726686 ms. Peak errors are
+0.705768 and +1.182352 mV. Duration-above--20-mV errors are -0.010208
and -0.003995 ms. The full trains contain 34 and 88 complete events,
versus 12 and 43 human events. Counts do not replace the retained event
times and intervals. This candidate is not physiologically qualified.

The conductance change causally alters the first minimum in this model.
It does not isolate the instantaneous Ca_LVA current as the only mediator:
calcium state, reversal voltage, and calcium-sensitive channels can also
change. Numerical refinement of this intervention remains open.

The [audit](h01-pv-calva-return-audit.json) retains all events, intervals,
first errors, and human minimum targets. Reproduce it with
`python -m docs.evidence.h01_pv_calva_return_audit`. Sixteen datum and
driver tests pass, including invalid factor rejection, valid zero, and
omitted defaults. Missing second positive spikes produce an invalid
comparison rather than a claimed improvement. The reserved trace was not read.
