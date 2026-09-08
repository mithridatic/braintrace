# E-gain cell usable tier

Model: SP3 gain-split candidate g1-ih-half (docs/specs/2026-09-07-h01-e-gain-split.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 110 pA | 0 | 0 | pass |
| 250 pA | 5 | 8 | FAIL |
| 310 pA | 10 | 10 | pass |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 110 pA | rate_hz |  | 0 | 0 |  |  | unavailable | no contract row | 15 percent of the human rate; no human repeats at this input |
| 110 pA | adaptation_ratio |  |  |  |  |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 250 pA | rate_hz |  | 4.79 | 7.57 | 0.718 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 250 pA | adaptation_ratio |  | 7.98 | 7.35 | 1.99 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 250 pA | width_ms | 1 | 0.936 | 0.996 | 0.187 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 2 | 1.13 | 0.995 | 0.227 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 3 | 0.975 | 0.987 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 4 | 0.971 | 0.987 | 0.194 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 5 | 0.963 | 0.986 | 0.193 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | ahp_mv | 1 | -69.6 | -70.3 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 2 | -67.1 | -70.8 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 3 | -70.2 | -71 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 4 | -70.3 | -71 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 5 | -70.6 | -71.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | rate_hz |  | 10 | 9.82 | 1.5 |  | pass | no contract row | 15 percent of the human rate; no human repeats at this input |
| 310 pA | adaptation_ratio |  | 9.95 | 8.56 | 2.49 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 310 pA | width_ms | 1 | 0.913 | 1.01 | 0.183 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 2 | 1.27 | 1.01 | 0.254 | 0.0118 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 3 | 1.09 | 0.99 | 0.219 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 4 | 0.986 | 0.987 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 5 | 0.988 | 0.986 | 0.198 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 6 | 0.977 | 0.986 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 7 | 0.984 | 0.985 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 8 | 0.979 | 0.985 | 0.196 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 9 | 0.967 | 0.985 | 0.193 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 10 | 0.965 | 0.985 | 0.193 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | ahp_mv | 1 | -69.6 | -69.9 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 2 | -65 | -70.9 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 3 | -67.6 | -70.8 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 4 | -69.2 | -70.9 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 5 | -69.5 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 6 | -69.6 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 7 | -69.4 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 8 | -69.6 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 9 | -69.8 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 10 | -69.9 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

### 110 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |

### 110 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |

### 250 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 59.98 | -69.56 | 0.936 | 36.03 | -55.84 |
| 2 | 34.26 | -67.09 | 1.135 | 35.31 | -54.63 |
| 3 | 220.12 | -70.22 | 0.975 | 34.88 | -54.44 |
| 4 | 307.42 | -70.28 | 0.971 | 34.78 | -54.06 |
| 5 | 273.28 | -70.59 | 0.963 | 34.56 | -54.31 |

### 250 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 65.86 | -70.34 | 0.996 | 37.76 | -57.21 |
| 2 | 21.69 | -70.83 | 0.995 | 37.77 | -57.24 |
| 3 | 49.51 | -70.95 | 0.987 | 37.14 | -57.21 |
| 4 | 190.45 | -71.05 | 0.987 | 37.15 | -57.21 |
| 5 | 174.10 | -71.10 | 0.986 | 37.14 | -57.21 |
| 6 | 167.37 | -71.14 | 0.986 | 37.14 | -57.21 |
| 7 | 162.68 | -71.16 | 0.985 | 37.13 | -57.21 |
| 8 | 159.47 | -71.18 | 0.985 | 37.13 | -57.21 |

### 310 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 38.02 | -69.56 | 0.913 | 35.34 | -56.41 |
| 2 | 12.14 | -64.97 | 1.269 | 33.56 | -55.03 |
| 3 | 60.86 | -67.63 | 1.094 | 34.63 | -53.88 |
| 4 | 109.76 | -69.16 | 0.986 | 33.91 | -53.25 |
| 5 | 115.30 | -69.50 | 0.988 | 33.66 | -52.78 |
| 6 | 119.20 | -69.56 | 0.977 | 33.34 | -52.94 |
| 7 | 122.14 | -69.44 | 0.984 | 33.34 | -53.44 |
| 8 | 116.16 | -69.63 | 0.979 | 33.00 | -52.88 |
| 9 | 123.54 | -69.84 | 0.967 | 32.75 | -53.19 |
| 10 | 120.76 | -69.91 | 0.965 | 32.69 | -54.28 |

### 310 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 41.86 | -69.87 | 1.006 | 38.44 | -57.24 |
| 2 | 13.65 | -70.90 | 1.011 | 39.00 | -57.35 |
| 3 | 26.91 | -70.82 | 0.990 | 37.34 | -57.23 |
| 4 | 132.37 | -70.93 | 0.987 | 37.10 | -57.22 |
| 5 | 135.11 | -70.99 | 0.986 | 37.11 | -57.22 |
| 6 | 128.03 | -71.03 | 0.986 | 37.10 | -57.22 |
| 7 | 124.03 | -71.05 | 0.985 | 37.09 | -57.22 |
| 8 | 120.96 | -71.08 | 0.985 | 37.09 | -57.22 |
| 9 | 118.67 | -71.09 | 0.985 | 37.09 | -57.22 |
| 10 | 116.93 | -71.11 | 0.985 | 37.08 | -57.22 |
