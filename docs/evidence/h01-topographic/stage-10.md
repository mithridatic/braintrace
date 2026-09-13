# SP15 stage 10: the coupling between the soma and the site

Every channel held; the axon stub's diameter changed (source 1 um), with the axonal NaTs density divided by the area factor in the total-held arms. Spike-1 take-off = first 10 V/s crossing at each point; approach measured over up to 10 ms before the somatic threshold.

## control 1 um (diameter 1.0 um, NaTs axon 3.814 S/cm2)

| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span mV | axon1 lead ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | read | 6 | 0.37 | -57.1 | -55.7 | -54.7 | 3.9 | +0.30 |
| 11.0 | read | 8 | 0.96 | -57.2 | -56.5 | -55.8 | 3.4 | +0.30 |
| 110.0 | read | 3 | 2.53 | -57.2 | -58.1 | -58.8 | 3.2 | +0.32 |

Slide, slowest to fastest ramp (0.37 to 2.53 mV/ms): soma -0.07 mV; axon0 -2.37 mV; axon1 -4.07 mV

310 pA step: not run

## x2 total held (diameter 2.0 um, NaTs axon 1.907 S/cm2)

| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span mV | axon1 lead ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | read | 6 | 0.51 | -53.1 | -53.1 | -52.7 | 6.1 | +0.16 |
| 11.0 | read | 8 | 1.04 | -53.8 | -54.0 | -54.0 | 7.5 | +0.18 |
| 110.0 | read | 3 | 2.57 | -56.0 | -56.5 | -56.9 | 8.9 | +0.14 |

Slide, slowest to fastest ramp (0.51 to 2.57 mV/ms): soma -2.87 mV; axon0 -3.44 mV; axon1 -4.24 mV

310 pA step: 10 spikes

| spike | approach | soma take-off | axon0 | axon1 | soma span | axon1 lead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.66 | -53.4 | -53.2 | -53.0 | 7.0 | +0.16 |
| 2 | 1.04 | -53.5 | -53.3 | -53.1 | 7.7 | +0.16 |
| 3 | 0.50 | -53.2 | -52.9 | -52.6 | 7.6 | +0.16 |
| 4 | 0.42 | -53.0 | -52.8 | -52.6 | 6.5 | +0.18 |
| 5 | 0.43 | -53.0 | -52.7 | -52.4 | 5.9 | +0.16 |
| 10 | 0.42 | -53.1 | -52.7 | -52.4 | 7.7 | +0.16 |

## x2 density held (diameter 2.0 um, NaTs axon 3.814 S/cm2)

| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span mV | axon1 lead ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | read | 5 | 0.45 | -55.2 | -54.8 | -54.4 | 5.3 | +0.20 |
| 11.0 | read | 8 | 0.99 | -55.7 | -55.6 | -55.5 | 4.1 | +0.20 |
| 110.0 | read | 3 | 2.50 | -57.3 | -57.8 | -58.2 | 6.0 | +0.18 |

Slide, slowest to fastest ramp (0.45 to 2.50 mV/ms): soma -2.08 mV; axon0 -3.00 mV; axon1 -3.77 mV

310 pA step: 9 spikes

| spike | approach | soma take-off | axon0 | axon1 | soma span | axon1 lead |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.63 | -55.5 | -55.1 | -54.6 | 4.1 | +0.18 |
| 2 | 1.14 | -55.5 | -55.2 | -54.7 | 4.5 | +0.18 |
| 3 | 0.41 | -55.1 | -54.7 | -54.2 | 5.4 | +0.20 |
| 4 | 0.37 | -55.2 | -54.8 | -54.1 | 4.2 | +0.18 |
| 5 | 0.37 | -55.1 | -54.6 | -54.2 | 5.3 | +0.20 |
| 9 | 0.36 | -55.1 | -54.6 | -54.1 | 3.8 | +0.20 |

## x4 total held (diameter 4.0 um, NaTs axon 0.9535 S/cm2)

| ramp pA/ms | status | count | approach mV/ms | soma take-off | axon0 take-off | axon1 take-off | soma span mV | axon1 lead ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | read | 5 | 0.54 | -52.2 | -52.2 | -52.2 | 11.3 | +0.04 |
| 11.0 | read | 7 | 1.04 | -53.0 | -53.1 | -53.1 | 11.5 | +0.04 |
| 110.0 | read | 3 | 2.60 | -55.7 | -55.9 | -56.0 | 10.7 | +0.04 |

Slide, slowest to fastest ramp (0.54 to 2.60 mV/ms): soma -3.44 mV; axon0 -3.62 mV; axon1 -3.81 mV

310 pA step: not registered

## Decision

- x4 total held PASS: somatic take-off falls by more than 1 mV from the slowest to the fastest ramp (-3.44)
- x4 total held PASS: axon[1] lead below 0.15 ms (+0.04)
- x2 total held PASS: somatic take-off falls by more than 0.5 mV (-2.87)
- x2 density held PASS: somatic take-off falls by more than 0.5 mV (-2.08)
- x2 arms FAIL: the two x2 slides agree within 0.5 mV (coupling, not density) (-0.79)
- x2 total held FAIL: somatic onset span stays within 3.5 to 5.7 mV (+8.93)
- x2 total held PASS: 310 pA count within 2 of 10 (+10.00)
- x2 density held FAIL: somatic onset span stays within 3.5 to 5.7 mV (+6.01)
- x2 density held PASS: 310 pA count within 2 of 10 (+9.00)
- x4 total held FAIL: somatic onset span stays within 3.5 to 5.7 mV (+11.49)

Verdict: FAIL

## Reading

The coupling makes the somatic take-off a function of the approach. Slowest to fastest ramp (about 0.4 to 2.6 mV/ms): control -0.07 mV; x2 with the total axonal sodium held -2.9 mV; x2 with the density held -2.1 mV; x4 with the total held -3.4 mV. The recorded soma slides -2.2 mV from 0.2 to 0.75 mV/ms. The axon lead falls from 0.30 ms to 0.16, 0.20 and 0.04 ms.
Two registered rejections fire by the letter. The two x2 arms differ by 0.8 mV (limit 0.5): the density modulates the slide, but both arms slide by thirty times the control's, so the coupling is the main effect and the density a secondary one. The somatic onset span leaves the recorded 3.5 to 5.7 mV band in every arm: 6.1 to 8.9 mV at x2 with the total held, 10.7 to 11.5 at x4, where the site has merged into the soma and the spike is becoming the whole-cell turnover of the I fit; at x2 with the density held the span is 4.1, 5.3 and 6.0 mV under the ramps and 3.8 to 5.4 along the 310 pA train, outside the band by 0.3 mV at one ramp only. The coupling therefore changes the spike as well as its reading, least in the density-held x2 arm.
The density-held x2 arm is the closest of anything this campaign has run to the recorded take-off: -55.2, -55.7 and -57.3 mV at 0.45, 0.99 and 2.50 mV/ms against the recording's -54.5 to -55.2 at 0.2 mV/ms and -57.2 at 0.75; span 4.1 to 6.0 against 3.5 to 5.7; 310 pA count 9 against 10. It does not reproduce the two other recorded elements: along the train the take-off holds at -55.1 to -55.5 mV (the recording steps up 1.9 mV after the first spike), and the spike-1 rise is 655 V/s against 348 (the x2 total-held arm 579), so the coupling is not the rise lever either.
Reading: the recorded soma reads its initiation site through a coupling tighter than the Allen stub, and the approach signature of the take-off follows that input and not a channel. What remains of the recorded take-off is what a spike leaves behind, +1.9 mV that no arm produces, and the rise, which no coupling touches.
