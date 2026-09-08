# Calcium response-time screen

The low-input audit compares complete positive-spike intervals at 0.19 nA.
The human third interval is 42.759876 ms. The selected human late interval
is 98.753820 ms. All candidates use the same phase build and other settings.

| Axonal gamma | Removal time, ms | Third interval, ms | Late interval, ms | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0.002 | 1000 | 11.925901 | 157.811711 | Control |
| 0.004 | 300 | 16.996475 | 87.761797 | Both absolute errors decrease |
| 0.008 | 150 | 105.504733 | 83.850993 | Third-interval error increases |

The predeclared screen selects gamma 0.004 and removal time 300 ms for
the 0.27 nA check. It does not validate this candidate. Its first interval
error increases from 1.429057 to 1.622804 ms. Its third interval still lacks
most of the observed human pause. Its late interval is now too short.

Both new candidates have the same gamma-times-removal-time product, but
their third intervals differ. Equal stationary calcium gain for an imposed
constant current does not imply equal firing in the coupled model.
The intervention changes both parameters. It does not identify their separate
effects or establish that SK alone causes the timing change.

The [audit](h01-pv-calcium-response-audit.json) retains all detected events,
the first three interval errors, first-spike errors, and selected late pair.
The four-interval RSS is a diagnostic score, not physiological qualification.
The reserved input was not used.

## Second input

The selected candidate was run at 0.27 nA with the same cell parameters.
The [second-input audit](h01-pv-calcium-response-high-audit.json) verifies
the matching parameters and retains all events and the direct errors.

The selected late interval changes from 40.585695 to 26.054280 ms. The
human interval is 25.259988 ms. The absolute error falls from 15.325707
to 0.794292 ms. The first three intervals are 4.987643, 4.255690, and
4.119400 ms, versus human 6.229948, 5.948006, and 6.880304 ms. Each error
decreases slightly, but all three intervals remain too short.

The first spike remains 0.667977 ms early, 1.497386 mV too low, and
0.046429 ms too short above -20 mV. Its shape is almost unchanged by
the calcium intervention. The candidate has 43 events, equal to the human
count, but the direct timing and shape still fail to reproduce the trace.
No physiological acceptance threshold was added after this comparison.
No candidate is promoted. Spatial refinement and BrainCell transfer remain
open. The next diagnosis must retain the low-input pause, the high-input
early intervals, and the first-spike shape as separate unresolved responses.

## Spatial limit

The selected high-input candidate fails the prospective two-mesh interval
screen. Refining every section from factor 9 to 27 changes the first three
intervals by 0.002149, 0.003529, and 0.007004 ms. Each is within the
0.05 ms diagnostic limit. The selected late interval changes by 1.844121 ms,
from 26.054280 to 27.898400 ms. This exceeds the 0.25 ms limit.

The event count changes from 43 to 41. The final ordinal pair differs by
69.906449 ms; two coarse events have no ordinal partner. Ordinal pairing
does not prove that paired late events represent the same trajectory event.
The first onset changes by 0.000402 ms, peak by -0.000982 mV, and duration
above -20 mV by less than 0.000001 ms.

The [mesh audit](h01-pv-calcium-response-mesh.json) verifies section and
parameter invariants, retains all events, and applies the original limits.
Run `python -m docs.evidence.h01_pv_calcium_response_mesh` to reproduce it.
Missing early or late positive-spike pairs invalidate the interval decision;
unmatched events remain explicit. No limit was relaxed after observing failure.

The early timing and first-shape discrepancies persist on these two meshes.
The late response is not numerically qualified. Its closer human interval
on either mesh is insufficient for promotion. Further spatial diagnosis must
precede claims of a calibrated full train.

## Third spatial level

Factor 27 to 81 passes the same narrow interval screen. The first three
interval changes are 0.000247, 0.000408, and 0.000826 ms. The selected late
interval changes by 0.240077 ms, within the declared 0.25 ms limit. The
factor-81 interval is 28.138478 ms, or 2.878490 ms longer than the human
interval. The earlier factor-9 to factor-27 failure remains valid.

The full train does not pass a convergence claim. The count changes from
41 to 40. The last common ordinal event shifts by 8.958165 ms. A narrow
interval pass must not replace this direct full-train evidence.

The first appreciable interval changes occur around the transition from
the initial burst: interval 7 to 8 changes by 0.386681 ms and interval
8 to 9 by 0.711390 ms. The subsequent differences accumulate. The exact
identity between onset differences and the sum of preceding interval
differences is verified to 1e-10 ms. This locates timing divergence in the
trace; it does not identify a unique channel or spatial region as its cause.

The [third-level audit](h01-pv-calcium-response-mesh-81.json) retains the
decision and all events. Run
`python -m docs.evidence.h01_pv_calcium_response_mesh --fine-factor 81`.
The [interval accumulation record](h01-pv-calcium-response-mesh-accumulation.json)
retains every common ordinal interval. Missing positive pairs and unmatched
events remain explicit limits. No model is promoted to BrainCell or a circuit.

## Time-integration split

On factor 27, changing CVode absolute tolerance from 1e-8 to 1e-10
preserves all 41 complete events. The largest ordinal onset change is
0.000068226 ms. The first three interval changes are 0.000000558,
-0.000000346, and 0.000001698 ms. The selected late interval changes
by 0.000003711 ms. All satisfy the prospective 0.01 ms limits.

The [tolerance audit](h01-pv-calcium-response-time.json) checks equal
geometry and cell parameters and retains every paired event, including
peak and duration changes. Run
`python -m docs.evidence.h01_pv_calcium_response_time`.
Missing positive pairs invalidate the interval decision; unequal counts
reject the screen even if the common events are close.

The tested tolerance effect is much smaller than the measured spatial
effect. This supports further spatial diagnosis. It does not prove an
exact solution, bound every possible time-integration error, or resolve
the full-train mesh discrepancy. The shared spike extraction helper's
two tests pass; this is supporting analysis verification, not model validation.
