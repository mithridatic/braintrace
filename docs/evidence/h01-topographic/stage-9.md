# SP15 stage 9: the take-off's own variables

## (a) E recording: later-spike residual against the preceding interval (retained traces)

Residual = take-off minus the spike-1 relation at the spike's own approach (stage 7).

| train | spike | interval ms | approach mV/ms | residual mV |
| --- | ---: | ---: | ---: | ---: |
| sweep 49 | 2 | 174 | 0.18 | +0.88 |
| sweep 49 | 3 | 734 | 0.19 | +0.93 |
| sweep 50 | 2 | 34 | 0.14 | +0.92 |
| sweep 50 | 3 | 220 | 0.20 | +1.16 |
| sweep 50 | 4 | 307 | 0.18 | +1.51 |
| sweep 50 | 5 | 273 | 0.20 | +1.28 |
| sweep 51 | 2 | 22 | 0.30 | +1.99 |
| sweep 51 | 3 | 132 | 0.22 | +1.29 |
| sweep 51 | 4 | 157 | 0.21 | +1.51 |
| sweep 51 | 5 | 170 | 0.18 | +0.80 |
| sweep 51 | 6 | 184 | 0.21 | +1.29 |
| sweep 51 | 7 | 185 | 0.23 | +1.90 |
| sweep 52 | 2 | 17 | 0.53 | +2.38 |
| sweep 52 | 3 | 88 | 0.25 | +1.92 |
| sweep 52 | 4 | 127 | 0.27 | +2.05 |
| sweep 52 | 5 | 129 | 0.23 | +1.53 |
| sweep 52 | 6 | 122 | 0.25 | +1.54 |
| sweep 52 | 7 | 131 | 0.23 | +1.87 |
| sweep 52 | 8 | 132 | 0.25 | +1.79 |
| sweep 52 | 9 | 145 | 0.22 | +1.58 |
| sweep 53 | 2 | 11 | 1.56 | +1.75 |
| sweep 53 | 3 | 61 | 0.22 | +1.73 |
| sweep 53 | 4 | 110 | 0.29 | +2.42 |
| sweep 53 | 5 | 115 | 0.31 | +2.90 |
| sweep 53 | 6 | 119 | 0.33 | +2.76 |
| sweep 53 | 7 | 122 | 0.29 | +2.23 |
| sweep 53 | 8 | 116 | 0.33 | +2.83 |
| sweep 53 | 9 | 124 | 0.32 | +2.51 |
| sweep 53 | 10 | 121 | 0.23 | +1.33 |
| sweep 55 | 2 | 11 | 1.64 | +1.91 |
| sweep 55 | 3 | 22 | 0.27 | +3.06 |
| sweep 55 | 4 | 78 | 0.30 | +2.02 |
| sweep 55 | 5 | 80 | 0.30 | +2.21 |
| sweep 55 | 6 | 86 | 0.27 | +2.12 |
| sweep 55 | 7 | 87 | 0.32 | +2.23 |
| sweep 55 | 8 | 87 | 0.29 | +2.29 |
| sweep 55 | 9 | 85 | 0.30 | +2.39 |
| sweep 55 | 10 | 93 | 0.32 | +3.14 |
| sweep 55 | 11 | 90 | 0.36 | +2.86 |
| sweep 55 | 12 | 91 | 0.29 | +2.23 |
| sweep 55 | 13 | 93 | 0.29 | +2.76 |

n 41; intervals 11 to 734 ms; slope -0.57 mV per decade of interval; short_under_30_ms median +1.99 mV (n 5); middle median +2.23 mV (n 13); long_over_100_ms median +1.54 mV (n 23); band spread 0.69 mV

## (b) E fit under current ramps: spike-1 take-off at three points

| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span | axon0 lead ms | axon1 lead ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | read | 6 | 0.37 | -57.1 | -55.7 | -54.7 | 3.9 | +0.16 | +0.30 |
| 11.0 | read | 8 | 0.96 | -57.2 | -56.5 | -55.8 | 3.4 | +0.18 | +0.30 |
| 110.0 | read | 3 | 2.53 | -57.2 | -58.1 | -58.8 | 3.2 | +0.20 | +0.32 |

Across 3 ramps, approach 0.37 to 2.53 mV/ms: soma take-off range 0.07 mV; axon0 take-off range 2.37 mV; axon1 take-off range 4.07 mV

## (c) I fit with axon probes, 0.19 nA: 10 V/s crossing order and span at five points

| spike | soma lead ms / take-off / span | axon0_0.1 lead ms / take-off / span | axon0_0.5 lead ms / take-off / span | axon0_0.9 lead ms / take-off / span | axon1_0.5 lead ms / take-off / span |
| ---: | --- | --- | --- | --- | --- |
| 1 | +0.00 / -60.5 / 13.5 | +0.02 / -60.8 / 14.5 | +0.06 / -61.3 / 14.0 | +0.06 / -61.6 / 13.3 | +0.08 / -62.3 / 12.8 |
| 2 | +0.00 / -60.5 / 13.6 | +0.00 / -60.6 / 13.7 | +0.00 / -61.6 / 13.8 | +0.00 / -62.4 / 13.2 | -0.04 / -63.4 / 12.5 |
| 3 | +0.00 / -60.5 / 12.9 | +0.00 / -60.8 / 12.8 | -0.04 / -62.0 / 13.0 | -0.06 / -63.0 / 13.2 | -0.12 / -64.4 / 12.3 |
| 4 | +0.00 / -60.5 / 12.8 | +0.00 / -60.8 / 12.6 | -0.06 / -62.5 / 11.5 | -0.10 / -63.6 / 12.9 | -0.18 / -65.4 / 10.6 |
| 14 | +0.00 / -60.5 / 14.4 | -0.02 / -60.7 / 13.8 | -0.08 / -62.9 / 12.1 | -0.12 / -64.3 / 10.9 | -0.22 / -66.3 / 9.9 |

14 spikes; largest lead over the soma: axon0_0.1 +0.02 ms, axon0_0.5 +0.06 ms, axon0_0.9 +0.06 ms, axon1_0.5 +0.08 ms; smallest span: soma 12.8 mV, axon0_0.1 12.2 mV, axon0_0.5 11.4 mV, axon0_0.9 10.3 mV, axon1_0.5 8.9 mV

## Decision

- E (a) fixed: recovering step: residual falls with interval by more than 3 sigma (0.75 mV) between short and long intervals (+0.69)
- E (b) FAIL: fit take-off at every point moves less than 1 mV across the ramps (+4.07)
- I (c) PASS: no probe leads the soma by more than 0.1 ms and every span exceeds 8 mV (+0.08; rejection fired by the letter: the earliest point at spike 1 is axon1_0.5 (lead +0.080 ms); the model's axon ends at axon[1], there is no point beyond it)

## Reading

(a) E recording: the later spikes sit above the spike-1 relation by +2.0 mV after intervals under 30 ms, +2.2 mV at 30 to 100 ms and +1.5 mV after intervals over 100 ms (up to 734 ms), a band spread of 0.69 mV against the 0.75 mV limit: a fixed step within the limit, with a downward trend of 0.6 mV per decade of interval that the limit does not resolve. What a spike leaves in the recorded take-off does not recover inside the pulse; stage 6 read that it has recovered by the next sweep.
(b) E fit under ramps, spike 1: the somatic take-off is -57.1 to -57.2 mV from 0.37 to 2.53 mV/ms (0.07 mV), as in the step pulses. The take-off at the axon points is not fixed: axon[1](0.5) falls from -54.7 to -58.8 mV (4.1 mV) and axon[0](0.5) from -55.7 to -58.1 mV (2.4 mV) as the approach quickens, the axon leading the soma by 0.30 ms at every rate. At the axon's take-off the soma sits at -58.3, -58.6 and -59.5 mV (1.2 mV range) while the axon is 3.6, 2.8 and 0.7 mV above it: under a slow approach the axon's own inward current carries it ahead of the soma before it takes off; under a fast one the electrode drives the soma and the axon lags. Prediction (b) fails at both axon points and holds at the soma.
So the fit does contain a take-off that slides with the approach, of the recorded sign and size (recorded soma: -2.2 mV from 0.2 to 0.75 mV/ms, -7 mV to the 7 mV/ms short square; fit axon: -4.1 mV from 0.37 to 2.53 mV/ms), at its initiation site; its soma does not show it because the soma is a load 45 um down a 1 um stub that reads the arrival of the axonal spike at one voltage. Along the retained trains the fit's axon take-off is a function of the approach only (-55.1 at the 310 pA onset, -54.3 at every later spike and at the slow 200 pA onset); the recorded +1.9 mV after a spike has no counterpart at either site.
(c) I fit with probes: no probe leads the soma by more than 0.08 ms (axon[1](0.5) at spike 1; later spikes soma first) and every span is 8.9 mV or more (soma 12.8 to 14.4): the whole 60 um axon and the soma turn over together. The registered rejection names axon[1](0.5) as the earliest point at spike 1, by 0.08 ms, four samples; the model's axon ends there, so there is no point to move outward to. The finalist's spike is a whole-cell turnover because its axon is a 60 um stub with no site that fires first; the stage-8 I reading is confirmed on a second run with five points.
Reading of the stage: in the E fit the recorded take-off signature lives at the initiation site and is hidden from the soma by the coupling between them; in the recording the soma shows it. The next split is an input split, registered as stage 10: hold every channel and change the coupling between the soma and the site.
