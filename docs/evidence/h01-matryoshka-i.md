# Matryoshka characterisation: inhibitory candidate

Source: [contract scorecard](h01-contract-scorecard.json), records `candidate x9`
at 0.19 and 0.27 nA, `candidate axon x2187`, `candidate x27`, and the source bias
pair. Every residual is model minus human. Families follow Hartshorne 2020
(scan pages p034, p099, p105): elemental, cyclical, structural, temporal, asked in
that order. Numerical decision limits are Dixon limits from three mesh repeats
(factor 1.81).

## Elemental: is the input specified?

No. Both I calibration recordings carry a 31.445 pA acquisition bias
([input datum](h01-pv-input-datum.json)); every I model residual to date used
bias 0. On the source model at 0.19 nA the bias alone changes the event count
from 11 to 22 and the first onset from +11.71 ms to -3.24 ms. That single input
change exceeds every candidate-family effect recorded on the Y3 page. Decision:
zero-bias I residuals are retired; the candidate must be re-run with the
recorded bias before any family is ranked. The rows below are therefore a
characterisation of the zero-bias model, kept as the pre-datum control.

## Cyclical: does the error repeat across inputs and events?

| Row | 0.19 nA | 0.27 nA | Reading |
| --- | ---: | ---: | --- |
| Event count | 15 vs 12 | 43 vs 43 at x9; 41 at x27; 40 at axon x2187 | Low input over-fires; high input count is mesh noise |
| First onset (ms) | +7.65 | -0.67 | Opposite sign: input-dependent, not a fixed offset |
| First peak (mV) | -1.95 | -1.50 | Shared sign and size |
| First duration above -20 mV (ms) | -0.052 | -0.046 | Shared; at the 0.05 allowance (ratio 1.0) |
| Recovery minima (mV) | +4.7 to +5.5 on every event | +6.1 to +8.4 on every event | Shared, uniform across events: real, second tier |
| Early intervals (ms) | +1.6, -0.2, -25.8 | -1.2, -1.7, -2.8, -7.7, -17.0, -15.9, -8.4 | Too short at high input, growing with event index |

## Structural: which rows move with the mesh?

At 0.27 nA the crossing rows from event 8 onward move by up to 139 ms across
x9 / x27 / x81 and the count moves 43 / 41 / 40: the late train is not resolved
at any tested mesh, so those rows are unrankable (verdict `unresolved`). Intervals
1 to 7, every recovery minimum, and every first-event row change by less than
0.02 ms or 0.02 mV across the same meshes: they are physical.

## Temporal: early versus late?

At 0.27 nA the interval error grows from -1.2 ms (interval 1) to -17.0 ms
(interval 5) and reverses to +6 ms by interval 8: adaptation is first too fast,
then too slow. At 0.19 nA the model fires through the human pause (interval 3
-25.8 ms). Both are the calcium and SK signature already conditional on the Y3
page (branch L3), consistent with family G3 as the temporal lever.

## Targets that survive the structural filter

- 0.27 nA intervals 3 to 7: -2.8 to -17.0 ms (ratio 1.4 to 8.5 against the
  derived 2 ms; events 5 to 7 crossings are targets at 14 to 47 ms).
- Recovery minima at both inputs: +5 to +8.4 mV (ratio 5 to 8).
- 0.19 nA: count 15 vs 12; first onset +7.65 ms; interval 3 -25.8 ms.
- First peak -1.5 to -1.9 mV (ratio 1.5 to 1.9): real but not a target.

Parked (ratio below 2): first duration, first rise-to-peak and peak-to-fall,
0.27 nA first onset, subthreshold samples (source model: 2 of 10 rows fail at
-0.11 nA, 7 of 10 at -0.05 nA, all by 1.0 to 1.95 mV; the candidate has no
passive run).

## Split this nominates

Stage 0 of the I campaign: candidate with recorded bias at both inputs, then the
mesh pair for decision limits. Stage A ranks families on the surviving rows only.
