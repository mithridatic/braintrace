# SP15 stage 0: Matryoshka and load-curve characterisation (retained traces)

## E cell

Repeat envelope (temporal family):

| sweep | count | first spike (ms) | threshold (mV) | AHP (mV) | max rise (V/s) | late level p50 (mV) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 56 | 1 | 205.2 | -54.719 | -70.000 | 332.0 | -67.063 |
| 59 | 1 | 208.1 | -55.156 | -69.531 | 328.1 | -66.813 |
| 60 | 1 | 152.0 | -54.750 | -69.625 | 329.7 | -66.750 |
| 61 | 1 | 198.5 | -54.500 | -69.875 | 328.9 | -66.281 |
| 62 | 1 | 203.3 | -54.969 | -70.219 | 328.9 | -66.500 |
| sigma | 0.000 | 23.427 | 0.251 | 0.279 | 1.503 | 0.300 |

Family contrasts (model minus human, in units of the repeat sigma):

| family | row | human | model | diff | ratio to sigma |
| --- | --- | ---: | ---: | ---: | ---: |
| elemental | post-spike level p50 200 pA | -67.063 | -65.003 | 2.060 | 6.864 |
| elemental | count 200 pA | 1 | 4 | 3 | sigma 0 (identical in every repeat) |
| elemental | first spike ms 200 pA | 205.2 | 126.9 | -78.300 | -3.342 |
| elemental | max rise of spike 1 (V/s) 200 pA | 332.0 | 597.7 | 265.6 | 176.8 |
| cyclical | threshold climb first to last 310 pA | 2.125 | 0.083 | -2.042 | -8.124 |
| cyclical | cycle 2 ms 310 pA | 12.140 | 13.480 | 1.340 | 0.105 |
| temporal | repeat sigma (first spike ms / threshold mV / AHP mV / level mV) | 23.427, 0.251, 0.279, 0.300 | n/a | n/a | 1.000 |

Effective capacitance from the onset slope: human 127.8 pF, model 124.5 pF (initial slope over 1 ms after pulse onset at the first input; effective near-soma capacitance, same estimator on both traces).

Load curve after spike 1 (median I_inj - C dV/dt by voltage bin, nA outward positive):

| who | window | V bin (mV) | I_load p50 (nA) | samples |
| --- | --- | --- | ---: | ---: |
| human | after spike 1 | [-75.0, -70.0] | n/a | 1 |
| human | after spike 1 | [-70.0, -65.0] | 0.196 | 38924 |
| human | after spike 1 | [-65.0, -60.0] | 0.216 | 588 |
| model | after spike 1 | [-75.0, -70.0] | 0.088 | 45 |
| model | after spike 1 | [-70.0, -65.0] | 0.107 | 322 |
| model | after spike 1 | [-65.0, -60.0] | 0.188 | 2714 |

## I cell

Repeat envelope (temporal family):

| sweep | count | first spike (ms) | threshold (mV) | AHP (mV) | max rise (V/s) | late level p50 (mV) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 1 | 67.200 | -58.531 | -78.844 | 576.6 | -75.344 |
| 41 | 1 | 84.180 | -57.938 | -79.219 | 565.6 | -75.531 |
| 42 | 1 | 68.420 | -58.281 | -79.031 | 576.6 | -75.156 |
| 43 | 1 | 78.480 | -57.656 | -79.000 | 570.3 | -75.469 |
| sigma | 0.000 | 8.160 | 0.384 | 0.154 | 5.318 | 0.165 |

Family contrasts (model minus human, in units of the repeat sigma):

| family | row | human | model | diff | ratio to sigma |
| --- | --- | ---: | ---: | ---: | ---: |
| elemental | post-spike level p50 0.19 nA | -66.500 | -71.220 | -4.720 | -28.541 |
| elemental | count 0.19 nA | 12 | 14 | 2 | sigma 0 (identical in every repeat) |
| elemental | first spike ms 0.19 nA | 20.600 | 27.920 | 7.320 | 0.897 |
| elemental | max rise of spike 1 (V/s) 0.19 nA | 596.9 | 591.8 | -5.095 | -0.958 |
| cyclical | threshold climb first to last 0.27 nA | 4.750 | 0.855 | -3.895 | -10.146 |
| cyclical | cycle 2 ms 0.27 nA | 6.280 | 16.360 | 10.080 | 0.788 |
| temporal | repeat sigma (first spike ms / threshold mV / AHP mV / level mV) | 8.160, 0.384, 0.154, 0.165 | n/a | n/a | 1.000 |

Effective capacitance from the onset slope: human 69.7 pF, model 56.7 pF (initial slope over 1 ms after pulse onset at the first input; effective near-soma capacitance, same estimator on both traces).

Load curve after spike 1 (median I_inj - C dV/dt by voltage bin, nA outward positive):

| who | window | V bin (mV) | I_load p50 (nA) | samples |
| --- | --- | --- | ---: | ---: |
| human | after spike 1 | [-75.0, -70.0] | 0.014 | 99 |
| human | after spike 1 | [-70.0, -65.0] | 0.035 | 113 |
| human | after spike 1 | [-65.0, -60.0] | -0.011 | 70 |
| model | after spike 1 | [-75.0, -70.0] | 0.171 | 676 |
| model | after spike 1 | [-70.0, -65.0] | 0.162 | 375 |
| model | after spike 1 | [-65.0, -60.0] | -0.045 | 49 |

## Structural family (both cells against both fits)

| pattern | E | I |
| --- | --- | --- |
| rheobase (human above model) | + (1 vs 4 spikes at 200 pA) | + (late-rate zero 169 vs 144 pA) |
| late gain (human steeper) | + (1/5/10/13 vs 4/8/10/12) | + (0.39 vs 0.26 Hz/pA) |
| threshold climbs along the train (model fixed) | + (-56.4 to -52.8 mV at 310 pA) | + (usable-tier cycle tables) |
| post-spike slowing shrinks with input (model brake grows) | + (one spike then hold at 200 pA; 120 ms cycles at 310 pA) | + (late cycle 100 ms at 0.19, 25 ms at 0.27 nA) |

## Largest contrasts

- E: elemental / max rise of spike 1 (V/s) 200 pA: diff 265.6, 176.8 sigma.
- I: elemental / post-spike level p50 0.19 nA: diff -4.720, -28.541 sigma.

Decision: family and branch named; no lever named.

