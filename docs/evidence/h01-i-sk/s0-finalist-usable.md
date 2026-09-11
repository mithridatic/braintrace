# I cell usable tier

Model: candidate with somatic Kv3 close factor 2.0 (energetic search finalist). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 0.19 nA | 12 | 14 | FAIL |
| 0.27 nA | 43 | 37 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.19 nA | rate_hz |  | 12.2 | 14.6 | 1.83 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 0.19 nA | adaptation_ratio |  | 13 | 2.42 | 3.24 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 0.19 nA | width_ms | 1 | 0.273 | 0.221 | 0.0547 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 2 | 0.277 | 0.221 | 0.0554 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 3 | 0.278 | 0.221 | 0.0556 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 4 | 0.283 | 0.222 | 0.0566 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 5 | 0.288 | 0.222 | 0.0576 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 6 | 0.29 | 0.222 | 0.0581 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 7 | 0.291 | 0.222 | 0.0581 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 8 | 0.292 | 0.222 | 0.0584 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 9 | 0.292 | 0.222 | 0.0585 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 10 | 0.3 | 0.222 | 0.06 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 11 | 0.298 | 0.222 | 0.0595 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | width_ms | 12 | 0.299 | 0.222 | 0.0599 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.19 nA | ahp_mv | 1 | -78.8 | -80.1 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 2 | -78.7 | -80.5 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 3 | -78.9 | -80.6 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 4 | -79.4 | -80.7 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 5 | -79.4 | -80.7 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 6 | -79.4 | -80.7 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 7 | -79.6 | -80.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 8 | -79.3 | -80.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 9 | -79.5 | -80.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 10 | -79.1 | -80.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 11 | -79.4 | -80.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.19 nA | ahp_mv | 12 | -79.2 | -80.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | rate_hz |  | 43.6 | 36.6 | 6.55 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 0.27 nA | adaptation_ratio |  | 4.25 | 1.86 | 1.06 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 0.27 nA | width_ms | 1 | 0.267 | 0.221 | 0.0534 | 0.00605 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 2 | 0.274 | 0.221 | 0.0548 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 3 | 0.276 | 0.221 | 0.0552 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 4 | 0.277 | 0.222 | 0.0554 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 5 | 0.277 | 0.222 | 0.0554 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 6 | 0.276 | 0.223 | 0.0552 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 7 | 0.276 | 0.223 | 0.0552 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 8 | 0.276 | 0.223 | 0.0552 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 9 | 0.277 | 0.224 | 0.0555 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 10 | 0.277 | 0.224 | 0.0554 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 11 | 0.277 | 0.224 | 0.0554 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 12 | 0.277 | 0.224 | 0.0554 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 13 | 0.278 | 0.224 | 0.0556 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 14 | 0.276 | 0.224 | 0.0553 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 15 | 0.279 | 0.224 | 0.0558 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 16 | 0.279 | 0.224 | 0.0558 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 17 | 0.279 | 0.224 | 0.0558 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 18 | 0.28 | 0.224 | 0.0559 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 19 | 0.279 | 0.224 | 0.0558 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 20 | 0.279 | 0.224 | 0.0558 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 21 | 0.28 | 0.224 | 0.0559 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 22 | 0.28 | 0.224 | 0.056 | 0.00605 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 23 | 0.282 | 0.224 | 0.0564 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 24 | 0.281 | 0.224 | 0.0563 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 25 | 0.281 | 0.224 | 0.0562 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 26 | 0.281 | 0.224 | 0.0562 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 27 | 0.282 | 0.224 | 0.0564 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 28 | 0.283 | 0.224 | 0.0566 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 29 | 0.281 | 0.224 | 0.0562 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 30 | 0.282 | 0.224 | 0.0563 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 31 | 0.282 | 0.224 | 0.0563 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 32 | 0.283 | 0.224 | 0.0565 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 33 | 0.281 | 0.224 | 0.0563 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 34 | 0.283 | 0.224 | 0.0566 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 35 | 0.283 | 0.224 | 0.0566 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 36 | 0.282 | 0.224 | 0.0564 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | width_ms | 37 | 0.283 | 0.224 | 0.0565 | 0.00605 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 0.27 nA | ahp_mv | 1 | -78.9 | -79 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 2 | -78.1 | -79.3 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 3 | -78 | -79.6 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 4 | -78.4 | -79.8 | 2 | 0.551 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 5 | -79 | -79.8 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 6 | -79.4 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 7 | -79.8 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 8 | -79.8 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 9 | -79.7 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 10 | -80 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 11 | -79.8 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 12 | -79.7 | -79.9 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 13 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 14 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 15 | -79.9 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 16 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 17 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 18 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 19 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 20 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 21 | -79.7 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 22 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 23 | -79.9 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 24 | -79.7 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 25 | -79.9 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 26 | -80 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 27 | -79.6 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 28 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 29 | -79.6 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 30 | -79.5 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 31 | -79.5 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 32 | -79.9 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 33 | -79.8 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 34 | -79.9 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 35 | -79.7 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 36 | -79.7 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 0.27 nA | ahp_mv | 37 | -79.6 | -80 | 2 | 0.551 | pass | pass | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

### 0.19 nA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 21.68 | -78.84 | 0.273 | 19.56 | -60.50 |
| 2 | 7.62 | -78.72 | 0.277 | 18.59 | -59.88 |
| 3 | 10.18 | -78.91 | 0.278 | 18.62 | -58.53 |
| 4 | 42.84 | -79.44 | 0.283 | 18.00 | -56.66 |
| 5 | 67.58 | -79.38 | 0.288 | 17.09 | -54.66 |
| 6 | 78.00 | -79.44 | 0.290 | 16.72 | -53.69 |
| 7 | 83.92 | -79.63 | 0.291 | 16.62 | -53.72 |
| 8 | 95.82 | -79.34 | 0.292 | 15.84 | -54.31 |
| 9 | 103.50 | -79.47 | 0.292 | 16.31 | -54.16 |
| 10 | 197.00 | -79.12 | 0.300 | 14.22 | -51.72 |
| 11 | 117.14 | -79.41 | 0.298 | 14.97 | -53.03 |
| 12 | 98.84 | -79.25 | 0.299 | 14.78 | -52.03 |

### 0.19 nA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 31.19 | -80.13 | 0.221 | 17.62 | -60.70 |
| 2 | 34.76 | -80.54 | 0.221 | 17.53 | -60.58 |
| 3 | 39.16 | -80.65 | 0.221 | 17.44 | -60.57 |
| 4 | 45.06 | -80.69 | 0.222 | 17.36 | -60.53 |
| 5 | 53.61 | -80.72 | 0.222 | 17.29 | -60.49 |
| 6 | 64.37 | -80.74 | 0.222 | 17.26 | -60.45 |
| 7 | 73.94 | -80.76 | 0.222 | 17.25 | -60.43 |
| 8 | 79.29 | -80.77 | 0.222 | 17.25 | -60.44 |
| 9 | 81.52 | -80.78 | 0.222 | 17.26 | -60.46 |
| 10 | 82.48 | -80.79 | 0.222 | 17.27 | -60.45 |
| 11 | 83.04 | -80.79 | 0.222 | 17.27 | -60.44 |
| 12 | 83.47 | -80.80 | 0.222 | 17.28 | -60.42 |
| 13 | 83.88 | -80.80 | 0.222 | 17.28 | -60.48 |
| 14 | 84.25 | -80.81 | 0.222 | 17.29 | -60.47 |

### 0.27 nA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 12.68 | -78.91 | 0.267 | 19.44 | -61.66 |
| 2 | 6.28 | -78.12 | 0.274 | 17.31 | -60.00 |
| 3 | 5.94 | -78.03 | 0.276 | 16.94 | -59.66 |
| 4 | 6.96 | -78.38 | 0.277 | 17.47 | -59.59 |
| 5 | 12.08 | -78.97 | 0.277 | 17.56 | -57.50 |
| 6 | 22.06 | -79.38 | 0.276 | 18.00 | -58.41 |
| 7 | 22.62 | -79.75 | 0.276 | 18.03 | -58.16 |
| 8 | 22.52 | -79.78 | 0.276 | 18.09 | -58.41 |
| 9 | 22.80 | -79.72 | 0.277 | 18.00 | -58.28 |
| 10 | 22.24 | -79.97 | 0.277 | 18.12 | -59.00 |
| 11 | 23.16 | -79.84 | 0.277 | 18.00 | -58.69 |
| 12 | 23.58 | -79.66 | 0.277 | 17.78 | -58.00 |
| 13 | 23.74 | -79.78 | 0.278 | 18.00 | -58.28 |
| 14 | 22.20 | -79.75 | 0.276 | 18.06 | -58.53 |
| 15 | 22.94 | -79.88 | 0.279 | 18.00 | -57.97 |
| 16 | 23.06 | -79.81 | 0.279 | 18.00 | -58.12 |
| 17 | 24.00 | -79.81 | 0.279 | 17.69 | -57.31 |
| 18 | 24.40 | -79.78 | 0.280 | 17.56 | -57.34 |
| 19 | 24.54 | -79.84 | 0.279 | 17.66 | -58.28 |
| 20 | 24.32 | -79.78 | 0.279 | 17.84 | -58.38 |
| 21 | 25.40 | -79.72 | 0.280 | 17.59 | -57.62 |
| 22 | 25.80 | -79.75 | 0.280 | 17.66 | -56.84 |
| 23 | 25.62 | -79.94 | 0.282 | 17.63 | -57.41 |
| 24 | 26.06 | -79.66 | 0.281 | 17.50 | -56.72 |
| 25 | 25.68 | -79.88 | 0.281 | 17.44 | -57.66 |
| 26 | 26.04 | -79.97 | 0.281 | 17.69 | -57.91 |
| 27 | 25.84 | -79.56 | 0.282 | 17.66 | -57.22 |
| 28 | 26.34 | -79.75 | 0.283 | 17.63 | -57.00 |
| 29 | 25.36 | -79.63 | 0.281 | 17.69 | -57.91 |
| 30 | 25.98 | -79.53 | 0.282 | 17.84 | -57.72 |
| 31 | 24.66 | -79.53 | 0.282 | 17.69 | -57.56 |
| 32 | 24.36 | -79.88 | 0.283 | 17.72 | -58.50 |
| 33 | 25.02 | -79.81 | 0.281 | 17.75 | -57.53 |
| 34 | 24.84 | -79.91 | 0.283 | 18.00 | -57.78 |
| 35 | 25.30 | -79.72 | 0.283 | 17.78 | -57.47 |
| 36 | 25.42 | -79.72 | 0.282 | 17.75 | -57.03 |
| 37 | 25.34 | -79.59 | 0.283 | 18.00 | -57.84 |
| 38 | 23.00 | -79.84 | 0.281 | 18.00 | -61.22 |
| 39 | 24.18 | -79.75 | 0.283 | 17.66 | -57.91 |
| 40 | 25.12 | -79.59 | 0.284 | 17.59 | -57.28 |
| 41 | 26.08 | -79.66 | 0.283 | 17.50 | -57.03 |
| 42 | 25.00 | -79.78 | 0.283 | 17.78 | -57.72 |
| 43 | 26.68 | -79.75 | 0.283 | 17.50 | -56.91 |

### 0.27 nA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 13.62 | -78.99 | 0.221 | 17.94 | -61.13 |
| 2 | 16.37 | -79.30 | 0.221 | 17.81 | -61.05 |
| 3 | 17.06 | -79.61 | 0.221 | 17.69 | -60.96 |
| 4 | 17.87 | -79.77 | 0.222 | 17.59 | -60.90 |
| 5 | 18.69 | -79.84 | 0.222 | 17.47 | -60.85 |
| 6 | 19.65 | -79.87 | 0.223 | 17.36 | -60.74 |
| 7 | 20.75 | -79.89 | 0.223 | 17.25 | -60.65 |
| 8 | 21.96 | -79.90 | 0.223 | 17.15 | -60.57 |
| 9 | 23.27 | -79.92 | 0.224 | 17.06 | -60.52 |
| 10 | 24.61 | -79.93 | 0.224 | 16.98 | -60.44 |
| 11 | 25.88 | -79.94 | 0.224 | 16.93 | -60.40 |
| 12 | 27.00 | -79.95 | 0.224 | 16.89 | -60.38 |
| 13 | 27.92 | -79.96 | 0.224 | 16.86 | -60.36 |
| 14 | 28.64 | -79.96 | 0.224 | 16.84 | -60.34 |
| 15 | 29.17 | -79.97 | 0.224 | 16.83 | -60.32 |
| 16 | 29.55 | -79.97 | 0.224 | 16.82 | -60.34 |
| 17 | 29.79 | -79.97 | 0.224 | 16.82 | -60.36 |
| 18 | 29.97 | -79.98 | 0.224 | 16.81 | -60.35 |
| 19 | 30.10 | -79.98 | 0.224 | 16.81 | -60.36 |
| 20 | 30.16 | -79.98 | 0.224 | 16.81 | -60.35 |
| 21 | 30.23 | -79.98 | 0.224 | 16.82 | -60.35 |
| 22 | 30.27 | -79.99 | 0.224 | 16.82 | -60.36 |
| 23 | 30.28 | -79.99 | 0.224 | 16.82 | -60.33 |
| 24 | 30.31 | -79.99 | 0.224 | 16.82 | -60.37 |
| 25 | 30.32 | -79.99 | 0.224 | 16.82 | -60.32 |
| 26 | 30.34 | -79.99 | 0.224 | 16.82 | -60.35 |
| 27 | 30.33 | -79.99 | 0.224 | 16.83 | -60.37 |
| 28 | 30.34 | -79.99 | 0.224 | 16.83 | -60.33 |
| 29 | 30.35 | -80.00 | 0.224 | 16.83 | -60.36 |
| 30 | 30.36 | -80.00 | 0.224 | 16.83 | -60.34 |
| 31 | 30.36 | -80.00 | 0.224 | 16.83 | -60.34 |
| 32 | 30.36 | -80.00 | 0.224 | 16.84 | -60.34 |
| 33 | 30.37 | -80.00 | 0.224 | 16.84 | -60.36 |
| 34 | 30.37 | -80.00 | 0.224 | 16.84 | -60.32 |
| 35 | 30.37 | -80.00 | 0.224 | 16.84 | -60.36 |
| 36 | 30.38 | -80.00 | 0.224 | 16.84 | -60.34 |
| 37 | 30.39 | -80.00 | 0.224 | 16.84 | -60.35 |
