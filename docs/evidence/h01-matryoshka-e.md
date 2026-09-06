# Matryoshka characterisation: excitatory candidate

Source: [contract scorecard](h01-contract-scorecard.json), records `candidate`,
`source`, and `into-F3F5 (matrix base 000)` at sweep 50 (0.25 nA with recorded
bias) and sweep 43 (0.11 nA). Every residual is model minus human. Numerical
decision limits are Dixon limits from the CVode 1e-10 / 1e-11 pair (factor 2.95);
the largest is 0.0033 ms, so every failed row is physical.

## Elemental: is the input specified?

Yes. The recorded command (0.24999 nA) and bias (-3.71 pA) are applied; the
sweep-53 holdout is unopened.

## Cyclical: does the error repeat across inputs and events?

Only one active calibration input exists, so no Youden pair of active responses
can be drawn; the tolerance pair supplies the repeatability axis instead.

| Row | Candidate | F3+F5 base | Source | Reading |
| --- | ---: | ---: | ---: | --- |
| Event count | 5 | 5 | 7 | Count carried by F3 (calcium) |
| Intervals 1-4 (ms) | -8.3, +51.6, -0.2, +20.6 | +3.3, -0.1, -9.0, +10.5 | -6.6, -96.7, -97.3, -77.3 | Non-monotonic in the candidate |
| Recovery minima 1-4 (mV) | +0.06, -2.86, +0.16, +0.18 | -1.56, -4.36, -1.29, -1.25 | -1.5, -4.4, -1.3, -1.3 | F1/F2/F4 buy the minima |
| Rise-to-peak, all five events (ms) | -0.12 to -0.15 | -0.17 | -0.17 | Uniform over events: shape, not adaptation |
| Peak-to-fall, events 1-4 (ms) | +0.12 to +0.15 | +0.14 | +0.15 | Uniform over events |
| Subthreshold 1019-1120 ms (mV) | +0.5 to +0.9 | same | 0 to +1.8 | F5 fixes the onset region |
| Subthreshold 1520-2120 ms (mV) | +1.2 to +1.7 | same | +2.1 to +4.1 | Late return still fails the 1 mV row |

## Structural: which rows move with the solver?

None above 0.0033 ms or 0.0002 mV. The mesh was fixed at nseg x9 by the Stage 0
preservation decision of the family campaign.

## Temporal: early versus late?

The candidate's interval errors alternate in sign (-8, +52, 0, +21 ms): not a
single adaptation constant. The F3+F5 base holds every interval within 10.5 ms,
so the temporal error of the candidate is introduced by F1/F2/F4, which are
also what move the minima. The phase errors are the same on every event, so
they are elemental (a channel law or density), not temporal.

## Targets that survive the structural filter

- Intervals 2 and 4 (+51.6, +20.6 ms) and events 3 to 5 crossings (+44 to
  +64 ms): targets.
- Recovery minimum 2 (-2.86 mV; base -4.36 mV): real.
- Phases on every event (-0.12 to -0.15 ms and +0.12 to +0.15 ms, ratio 2.4 to
  3): real, elemental.
- Late subthreshold samples (+1.2 to +1.7 mV): real, and not addressed by the
  F1/F2/F4 split; carried as an open row.

Parked: event 1 onset (+0.90 ms), all peak voltages, minima 1, 3, 4, early
subthreshold samples.

## Split this nominates

The 2^3 matrix over F1, F2, F4 on the F3+F5 base, judged per group: minima
(1 mV) and crossings (1 ms) together. The late subthreshold row is scored on
every cell but is not expected to move; if it does not, it becomes the next
elemental question outside F1-F5.
