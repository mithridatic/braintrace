# SP15 stage 8: soft or hard take-off

Retained traces only, uniform 0.02 ms grid, rate of rise as a central difference over 0.1 ms. Onset read from the first 10 V/s crossing to the 100 V/s crossing: span (mV), rapidness (slope of dV/dt against V from 10 to 50 V/s, per ms). Fits also at their axon point (E axon[1](0.5), 45 um out on the 60 um stub; I axon[0](0.5), first axon section): lead of the axonal 10 V/s crossing over the somatic one, and the axon voltage when the soma crosses 10 V/s.

## E human

| train | spike | threshold | soma span | soma rapidness |
| --- | ---: | ---: | ---: | ---: |
| sweep 48 | 1 | -55.6 | 3.66 | 31 |
| sweep 49 | 1 | -56.2 | 4.94 | 36 |
| sweep 49 | 2 | -54.7 | 4.50 | 33 |
| sweep 49 | 3 | -54.7 | 5.62 | 31 |
| sweep 50 | 1 | -55.8 | 5.66 | 36 |
| sweep 50 | 2 | -54.6 | 6.00 | 29 |
| sweep 50 | 3 | -54.4 | 5.63 | 30 |
| sweep 50 | 4 | -54.1 | 5.00 | 33 |
| sweep 50 | 5 | -54.3 | 5.97 | 28 |
| sweep 51 | 1 | -57.3 | 3.53 | 34 |
| sweep 51 | 2 | -53.7 | 5.25 | 30 |
| sweep 51 | 3 | -54.3 | 5.00 | 32 |
| sweep 51 | 4 | -54.1 | 5.06 | 32 |
| sweep 51 | 5 | -54.8 | 4.44 | 32 |
| sweep 51 | 6 | -54.3 | 6.50 | 32 |
| sweep 51 | ... | | | | (1 more)
| sweep 52 | 1 | -56.6 | 4.56 | 36 |
| sweep 52 | 2 | -53.5 | 4.44 | 37 |
| sweep 52 | 3 | -53.7 | 4.59 | 31 |
| sweep 52 | 4 | -53.6 | 4.16 | 31 |
| sweep 52 | 5 | -54.1 | 5.81 | 28 |
| sweep 52 | 6 | -54.1 | 6.16 | 26 |
| sweep 52 | ... | | | | (3 more)
| sweep 53 | 1 | -56.4 | 4.50 | 39 |
| sweep 53 | 2 | -55.0 | 5.50 | 29 |
| sweep 53 | 3 | -53.9 | 4.50 | 32 |
| sweep 53 | 4 | -53.3 | 4.81 | 33 |
| sweep 53 | 5 | -52.8 | 5.19 | 29 |
| sweep 53 | 6 | -52.9 | 5.56 | 30 |
| sweep 53 | ... | | | | (4 more)
| sweep 55 | 1 | -57.2 | 3.72 | 34 |
| sweep 55 | 2 | -54.9 | 5.16 | 33 |
| sweep 55 | 3 | -52.6 | 5.16 | 28 |
| sweep 55 | 4 | -53.7 | 5.69 | 29 |
| sweep 55 | 5 | -53.5 | 5.28 | 29 |
| sweep 55 | 6 | -53.5 | 5.41 | 29 |
| sweep 55 | ... | | | | (7 more)
| sweep 56 | 1 | -54.7 | 5.66 | 32 |
| sweep 59 | 1 | -55.2 | 5.69 | 33 |
| sweep 60 | 1 | -54.8 | 5.31 | 35 |
| sweep 61 | 1 | -54.5 | 5.28 | 34 |
| sweep 62 | 1 | -55.0 | 5.09 | 33 |

## E fit

| train | spike | threshold | soma span | soma rapidness | axon span | axon lead ms | axon mV at soma 10 V/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 pA | 1 | -57.2 | 5.51 | 40 | 7.44 | 0.280 | -40.9 |
| 200 pA | 2 | -57.0 | 4.93 | 41 | 7.08 | 0.280 | -41.8 |
| 200 pA | 3 | -57.1 | 4.05 | 44 | 8.61 | 0.300 | -38.0 |
| 200 pA | 4 | -57.0 | 5.17 | 40 | 7.24 | 0.280 | -41.3 |
| 250 pA | 1 | -57.0 | 5.11 | 42 | 9.18 | 0.280 | -42.2 |
| 250 pA | 2 | -57.1 | 4.10 | 45 | 8.48 | 0.300 | -38.8 |
| 250 pA | 3 | -57.0 | 5.24 | 41 | 7.26 | 0.280 | -41.4 |
| 250 pA | 4 | -57.1 | 4.13 | 44 | 8.66 | 0.300 | -37.9 |
| 250 pA | 5 | -57.1 | 3.71 | 37 | 8.28 | 0.300 | -39.1 |
| 250 pA | 6 | -57.2 | 3.29 | 39 | 7.70 | 0.280 | -40.1 |
| 250 pA | ... | | | | | | | (2 more)
| 310 pA | 1 | -57.2 | 3.39 | 41 | 7.78 | 0.300 | -41.2 |
| 310 pA | 2 | -57.2 | 4.68 | 44 | 8.53 | 0.280 | -43.8 |
| 310 pA | 3 | -57.2 | 3.82 | 37 | 8.30 | 0.300 | -39.2 |
| 310 pA | 4 | -57.2 | 3.26 | 39 | 7.68 | 0.280 | -40.2 |
| 310 pA | 5 | -57.2 | 3.30 | 39 | 7.71 | 0.280 | -40.1 |
| 310 pA | 6 | -57.2 | 3.20 | 39 | 7.63 | 0.280 | -40.3 |
| 310 pA | ... | | | | | | | (4 more)

## I human

| train | spike | threshold | soma span | soma rapidness |
| --- | ---: | ---: | ---: | ---: |
| 0.19 nA | 1 | -60.5 | 1.88 | 51 |
| 0.19 nA | 2 | -59.9 | 1.88 | 51 |
| 0.19 nA | 3 | -58.5 | 3.53 | 48 |
| 0.19 nA | 4 | -56.7 | 3.59 | 45 |
| 0.19 nA | 5 | -54.7 | 4.50 | 36 |
| 0.19 nA | 6 | -53.7 | 3.84 | 43 |
| 0.19 nA | ... | | | | (6 more)
| 0.27 nA | 1 | -61.7 | 3.81 | 38 |
| 0.27 nA | 2 | -60.0 | 3.81 | 38 |
| 0.27 nA | 3 | -59.7 | 2.72 | 41 |
| 0.27 nA | 4 | -59.6 | 4.38 | 41 |
| 0.27 nA | 5 | -57.5 | 3.66 | 46 |
| 0.27 nA | 6 | -58.4 | 4.09 | 53 |
| 0.27 nA | ... | | | | (37 more)
| repeat sweep 40 | 1 | -58.5 | 3.22 | 53 |
| repeat sweep 41 | 1 | -57.9 | 3.12 | 52 |
| repeat sweep 42 | 1 | -58.3 | 2.06 | 58 |
| repeat sweep 43 | 1 | -57.7 | 2.34 | 56 |

## I fit

| train | spike | threshold | soma span | soma rapidness | axon span | axon lead ms | axon mV at soma 10 V/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.19 nA | 1 | -60.5 | 13.54 | 5 | 14.02 | 0.060 | -60.7 |
| 0.19 nA | 2 | -60.5 | 13.58 | 5 | 13.82 | 0.000 | -61.6 |
| 0.19 nA | 3 | -60.5 | 12.92 | 5 | 12.98 | -0.040 | -62.4 |
| 0.19 nA | 4 | -60.5 | 12.79 | 5 | 11.52 | -0.060 | -63.1 |
| 0.19 nA | 5 | -60.5 | 12.77 | 5 | 12.88 | -0.060 | -63.6 |
| 0.19 nA | 6 | -60.3 | 13.86 | 5 | 11.52 | -0.060 | -63.8 |
| 0.19 nA | ... | | | | | | | (8 more)
| 0.27 nA | 1 | -61.1 | 14.95 | 5 | 13.47 | 0.020 | -61.6 |
| 0.27 nA | 2 | -61.0 | 14.93 | 5 | 13.87 | -0.020 | -62.6 |
| 0.27 nA | 3 | -61.0 | 13.11 | 5 | 11.98 | -0.060 | -63.5 |
| 0.27 nA | 4 | -60.9 | 12.95 | 5 | 12.37 | -0.080 | -64.4 |
| 0.27 nA | 5 | -60.7 | 14.71 | 5 | 12.92 | -0.100 | -65.3 |
| 0.27 nA | 6 | -60.7 | 13.64 | 5 | 10.81 | -0.120 | -66.3 |
| 0.27 nA | ... | | | | | | | (31 more)

## Summary (median, min to max)

- E human, soma, spike 1: span 5.02 mV (3.53 to 5.69), rapidness 34/ms (31 to 39), n 12
- E human, soma, later: span 5.28 mV (4.16 to 6.50), rapidness 29/ms (25 to 37), n 41
- E fit, soma, spike 1: span 5.11 mV (3.39 to 5.51), rapidness 41/ms (40 to 42), n 3
- E fit, soma, later: span 4.10 mV (3.12 to 5.24), rapidness 41/ms (36 to 45), n 19
- E fit, axon, spike 1: span 7.78 mV (7.44 to 9.18), rapidness 9/ms (9 to 10), n 3
- E fit, axon, later: span 8.28 mV (7.08 to 8.99), rapidness 10/ms (9 to 10), n 19
- I human, soma, spike 1: span 2.73 mV (1.88 to 3.81), rapidness 52/ms (38 to 58), n 6
- I human, soma, later: span 3.63 mV (1.88 to 4.72), rapidness 43/ms (28 to 69), n 53
- I fit, soma, spike 1: span 14.24 mV (13.54 to 14.95), rapidness 5/ms (5 to 5), n 2
- I fit, soma, later: span 13.35 mV (12.76 to 14.93), rapidness 5/ms (5 to 5), n 49
- I fit, axon, spike 1: span 13.75 mV (13.47 to 14.02), rapidness 5/ms (5 to 5), n 2
- I fit, axon, later: span 10.10 mV (9.04 to 13.87), rapidness 7/ms (5 to 8), n 49

## Axon lead (fits)

- E fit: n 22; lead 0.280 ms median, 0.280 min; axon first at every spike: True; axon at -40.4 mV (median over spikes) when the soma crosses 10 V/s; spike 1 of each train: lead +0.280 ms, axon -40.9 mV; lead +0.280 ms, axon -42.2 mV; lead +0.300 ms, axon -41.2 mV
- I fit: n 51; lead -0.180 ms median, -0.180 min; axon first at every spike: False; axon at -69.8 mV (median over spikes) when the soma crosses 10 V/s; spike 1 of each train: lead +0.060 ms, axon -60.7 mV; lead +0.020 ms, axon -61.6 mV

## The E fit's take-off at the soma and at the axon point under the SP16 gate (310 pA, retained stage-1 traces)

| gate depth | spike | soma 10 V/s mV | axon 10 V/s mV | axon lead ms |
| --- | ---: | ---: | ---: | ---: |
| depth 0 | 1 | -57.2 | -55.1 | 0.30 |
| depth 0 | 2 | -57.4 | -55.2 | 0.28 |
| depth 0 | 5 | -57.2 | -54.3 | 0.28 |
| depth 0 | 10 | -57.1 | -54.4 | 0.30 |
| depth 0.3 | 1 | -57.1 | -55.0 | 0.30 |
| depth 0.3 | 2 | -57.1 | -55.1 | 0.30 |
| depth 0.3 | 5 | -56.7 | -54.0 | 0.30 |
| depth 0.3 | 10 | -56.6 | -53.7 | 0.28 |
| depth 0.6 | 1 | -57.2 | -55.0 | 0.28 |
| depth 0.6 | 2 | -56.9 | -54.8 | 0.30 |
| depth 0.6 | 5 | -56.2 | -53.5 | 0.30 |
| depth 0.6 | 10 | -55.7 | -52.9 | 0.30 |

## Repeat spread of spike 1

- E: n 5; span sd 0.26 mV (range 0.59); rapidness sd 1.2/ms
- I: n 4; span sd 0.57 mV (range 1.16); rapidness sd 3.0/ms

## Decision

- E FAIL: fit somatic onset is a kink (span under 1 mV) at every spike (+5.51)
- E PASS: fit axon point crosses 10 V/s before the soma at every spike (+0.28)
- E PASS: recording spike-1 span is resolved (not 1 to 2 mV) and its repeats spread under 1 mV (+5.02)
- E FAIL: recording and fit spike-1 spans differ by more than 3 sigma of the repeat spread (-0.09, 0.4 sigma)
- E PASS: E recording spike-1 onset is gradual (span over 2 mV) (+5.02)
- I FAIL: fit somatic onset is a kink (span under 1 mV) at every spike (+14.95)
- I FAIL: fit axon point crosses 10 V/s before the soma at every spike (-0.18)
- I FAIL: recording spike-1 span is resolved (not 1 to 2 mV) and its repeats spread under 1 mV (+2.73)
- I PASS: recording and fit spike-1 spans differ by more than 3 sigma of the repeat spread (-11.51, 20.1 sigma)

Classes: E recording gradual (5.0 mV), fit gradual (5.1 mV), fit/recording 1.0; I recording gradual (2.7 mV), fit gradual (14.2 mV), fit/recording 5.2; the absolute bands were registered for the E cell and are not applied to I, which is read as the ratio
Registered rejection: E not fired; I fired: repeat span range 1.16 mV over 1 mV; the reading is provisional
Verdict: E FAIL, I FAIL

## Reading

E: the somatic onset has the same shape in the recording and in the fit. From 10 to 100 V/s the recording covers 5.0 mV (3.5 to 5.7 over 12 trains, repeat sd 0.26) and the fit 5.1 mV (3.4 to 5.5); the rapidness is 34 against 41 per ms. Both are gradual by the registered class, and they differ by 0.4 sigma in span. Yet in the fit the axon point crosses 10 V/s 0.28 ms before the soma at every one of 22 spikes and is at about -40 mV, mid-spike, when the soma starts: the somatic onset of this fit is driven by an axonal spike that has already fired, and it still turns over as gradually as the recording. Prediction 1 is refuted: an imposed take-off does not show as a kink at this soma. The onset shape therefore does not discriminate the two take-off mechanisms that stage 7 separated; what differs is the voltage at which the turnover begins, fixed in the fit, trajectory-set in the recording.
I (provisional: the registered rejection fired, the four repeats spread 1.16 mV in span). The recording's onset covers 2.7 mV from 10 to 100 V/s (1.9 to 3.8; rapidness 52 per ms), gradual by the absolute band registered for the E cell but a fifth of its fit's, which turns over across 14.2 mV (rapidness 5 per ms): 20 sigma of the repeat spread and ten times its range. The reading stands on that ratio, not on the band, until a repeat set resolves the span; stage 9's I probe run serves as that confirmation. In the fit the soma and the first axon section rise together, the soma crossing 10 V/s 0.18 ms before the axon point and the axon point overtaking it only above 50 V/s; the earlier axon-first readings were taken at a high fixed voltage and describe the peak, not the onset. The I fit has no sharp initiation: its spike begins as a whole-cell turnover, where the recorded cell's begins five times more abruptly, as a spike arriving from a site that fires first. This is the largest elemental contrast in the I cell after the count, and it is a contrast of the initiation site, an input.
E fit, the take-off at the axon point: -54.3 to -55.2 mV across every spike and drive, sliding 0.9 mV from 0.25 to 1.16 mV/ms while the soma holds at -57.2; under the SP16 gate at depth 0.6 the axon point's take-off climbs 2.1 mV by spike 10 (-55.0 to -52.9) and the soma's 1.5 mV, together, with the lead unchanged at 0.3 ms. So the fit's take-off is set at or proximal to the axon point (45 um out; the point fires first, so initiation is at or before it), and there it answers to a 40 percent loss of sodium availability by only 2 mV: a dense insertion with fast kinetics crosses its own threshold at nearly the same voltage whatever its availability. The recording moves 2.2 mV with the approach alone and 1.9 mV more after one spike, which is more than availability at such a site can give.
Both-gradual was not a registered reading for E; it is recorded as a no-discrimination outcome of this split, not as evidence for either branch. The next E split has to act on the timing of the axonal spike relative to the somatic approach, which is where the fixed take-off is made; the next I split is the initiation site's density, length and coupling.
