# SP15 stage 13: the base as a sub-system (HL23PYR beside B3 and the recording)

Every model quantity is read through the recording's chain (stage 12: pipette pole 4.7 us, corner 10 kHz).

| drive nA | system | count | rest mV | spike-1 rise | fall | peak | threshold | take-off | approach | span |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | HL23PYR | 16 | -74.0 | 592 | -79 | 32.1 | -60.00 | -60.00 | 1.09 | 5.40 |
| 0.2 | B3 | 4 | -84.0 | 545 | -94 | 36.8 | -57.08 | -57.30 | 0.25 | 4.28 |
| 0.2 | recording | 1 | -83.8 | 332 | -105 | 36.2 | -54.72 | -54.94 | 0.22 | 5.66 |
| 0.25 | HL23PYR | 18 | -74.5 | 594 | -80 | 32.1 | -59.90 | -59.90 | 1.40 | 4.33 |
| 0.25 | B3 | 8 | -84.0 | 556 | -94 | 37.5 | -57.14 | -57.14 | 0.40 | 3.76 |
| 0.25 | recording | 5 | -84.3 | 351 | -105 | 36.0 | -55.84 | -55.84 | 0.44 | 5.66 |
| 0.31 | HL23PYR | 20 | -74.5 | 600 | -80 | 32.1 | -60.07 | -60.07 | 1.85 | 4.97 |
| 0.31 | B3 | 10 | -84.0 | 570 | -94 | 38.2 | -57.29 | -57.29 | 0.58 | 4.80 |
| 0.31 | recording | 10 | -83.9 | 348 | -104 | 35.3 | -56.41 | -56.41 | 0.63 | 4.50 |
| 0.4 | HL23PYR | 24 | -74.5 | 600 | -80 | 32.4 | -60.21 | -74.22 | 2.59 | 19.97 |

## Threshold along the train (cycle by cycle, spikes 1 to 6)

- HL23PYR 0.2 nA: -60.00, -59.23, -59.17, -59.01, -59.10, -58.92 mV; spike 2 +0.77, spike 5 +0.90; rise 5/1 0.97
- HL23PYR 0.25 nA: -59.90, -59.02, -58.92, -58.87, -58.72, -58.78 mV; spike 2 +0.88, spike 5 +1.18; rise 5/1 0.97
- HL23PYR 0.31 nA: -60.07, -58.69, -58.83, -58.44, -58.29, -58.30 mV; spike 2 +1.38, spike 5 +1.78; rise 5/1 0.95
- HL23PYR 0.4 nA: -60.21, -57.97, -58.33, -58.03, -57.85, -57.85 mV; spike 2 +2.24, spike 5 +2.36; rise 5/1 0.94
- B3 0.25 nA: -57.14, -57.24, -57.12, -57.20, -57.24, -57.05 mV; spike 2 -0.10, spike 5 -0.10; rise 5/1 0.97
- B3 0.31 nA: -57.29, -57.32, -57.26, -57.06, -57.06, -57.07 mV; spike 2 -0.02, spike 5 +0.24; rise 5/1 0.95
- recording 0.25 nA: -55.84, -54.63, -54.44, -54.06, -54.31 mV; spike 2 +1.22, spike 5 +1.53; rise 5/1 0.88
- recording 0.31 nA: -56.41, -55.03, -53.88, -53.25, -52.78, -52.94 mV; spike 2 +1.38, spike 5 +3.63; rise 5/1 0.86

HL23PYR spike-1 take-off against approach: {'n': 3, 'approach_range': [1.0895273033797859, 1.8514908769456366], 'take_off_range': [-60.068396259297316, -59.90368684909536], 'slope_mv_per_mv_ms': -0.10891273104305527, 'slope_mv_per_decade': -0.31458229900789925, 'overlaps_recorded_range': False, 'excluded_out_of_band_spans': 1}

## Decision

- (a) FAIL: spike-1 rise through the chain within 15 percent of 348 V/s (lowest firing drive 0.2 nA) (+592; at the matched drive 0.31 nA 600; B3 through the chain 570)
- (b) FAIL: threshold climbs at least half the recorded +1.4 (spike 2) and +3.6 mV (spike 5), 0.2 nA train (+0.896; spike 2 +0.77, spike 5 +0.90 mV; at the matched drive 0.31 nA +1.38 / +1.78; B3 -0.02 / +0.24)
- (b) pass: rise of spike 5 above 0.8 of spike 1 (+0.969; recorded 0.86; B3 0.95)
- (c) informational: spike-1 fall through the chain near the recorded -104 V/s (informational) (-79.5; B3 -94)
- (d) informational: spike-1 take-off against approach (informational): recorded -3.8 mV per decade over 0.2 to 0.75 mV/ms, B3 flat (-0.315; HL23PYR -0.31 mV per decade over 1.09 to 1.85 mV/ms, DISJOINT from the recorded range; 1 drive excluded for an out-of-band span (reader failure at a fast approach))
- (e) FAIL: 0.31 nA count within 5 to 15 (rise not bought with the count) (+20; recorded 10, B3 10; counts at 0.2/0.25/0.31/0.4: 16/18/20/24)

Verdict: FAIL

## Reading

- The rise is not the fitting pipeline. Two independently fitted human L2/3 models, B3 (Allen, fitted to this very cell's sweeps) and HL23PYR (Toronto, a population fit with human-shifted NaTg on a different anatomy), both rise at 570 to 600 V/s through the recording's chain against the recorded 348, at every drive. What the two share is the sodium equation family (Colbert and Pan 2002, shifted and re-dosed) and the ideal current clamp; what differs between them (densities, shifts, anatomy, initial segment) does not move the rise. Replacing the base does not close that gap.
- The threshold climb has a positive control. HL23PYR climbs along its train, +1.38 mV at spike 2 and +1.78 by spike 5 at 0.31 nA (+2.24 / +2.36 at 0.4), where B3 gives -0.02 / +0.24 and the recording +1.38 / +3.63, with the rise kept (0.95 of spike 1; recorded 0.86). The registered 'at least half' fired by the letter on the lowest firing drive (0.2 nA: +0.77 / +0.90) and sits at the limit at the matched drive. The shape differs: HL23PYR's climb is a step in the first interval that then holds, the recording's is that step plus an accumulation of +2.2 mV more over the next three spikes. So a human-fitted base reproduces the step and not the accumulation.
- HL23PYR is the wrong cell in excitability: it rests at -74 mV where this cell rests at -84 under the same bias, fires 16, 18, 20 and 24 spikes where the recording fires 1, 5 and 10, and its 0.4 nA take-off reading is a reader failure at a 2.6 mV/ms approach (excluded). Its spike-1 take-off against approach is read over 1.1 to 1.9 mV/ms, disjoint from the recorded 0.2 to 0.75, and is not compared as a value. Its repolarisation, -80 V/s, is slower than B3's -94 and further from the recorded -104.
- The elemental picture, read from the same plot: through the chain the three spike-1 loops coincide from the take-off to about -40 mV (the recording, B3 and HL23PYR all pass 200 V/s near -45 mV), and separate above it, where the two fits accelerate to 550 to 575 V/s and the recording levels at 340. The fits' excess rise lives above -40 mV, in the phase the soma's own sodium carries, not in the phase delivered by the initiation site; the recording's repolarisation is faster and its loop rounder.
- Consequence for the search: the base stays B3; the second half (HL23PYR biophysics on the 541563728 anatomy) is worth one run only for the climb, not for the rise. The rise stays the steepest element and now has the same value on two bases that differ in everything but their sodium equations, which is the sub-system the next split has to swap or dose as a characteristic curve, read through the chain.
