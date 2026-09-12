# E-gain cell usable tier

Model: SP3 gain-split candidate s1-ksahp-tail5 (docs/specs/2026-09-07-h01-e-gain-split.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 250 pA | 5 | 3 | FAIL |
| 310 pA | 10 | 7 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 pA | rate_hz |  | 4.79 | 4.1 | 0.718 |  | pass | no contract row | 15 percent of the human rate; no human repeats at this input |
| 250 pA | adaptation_ratio |  | 7.98 | 9.69 | 1.99 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 250 pA | width_ms | 1 | 0.936 | 0.996 | 0.187 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 2 | 1.13 | 0.987 | 0.227 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 3 | 0.975 | 0.983 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | ahp_mv | 1 | -69.6 | -70.7 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 2 | -67.1 | -70.9 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 3 | -70.2 | -71 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | rate_hz |  | 10 | 6.71 | 1.5 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 310 pA | adaptation_ratio |  | 9.95 | 9.97 | 2.49 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 310 pA | width_ms | 1 | 0.913 | 1.01 | 0.183 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 2 | 1.27 | 0.998 | 0.254 | 0.0118 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 3 | 1.09 | 0.985 | 0.219 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 4 | 0.986 | 0.985 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 5 | 0.988 | 0.984 | 0.198 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 6 | 0.977 | 0.984 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 7 | 0.984 | 0.983 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | ahp_mv | 1 | -69.6 | -70.3 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 2 | -65 | -70.8 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 3 | -67.6 | -70.9 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 4 | -69.2 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 5 | -69.5 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 6 | -69.6 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 7 | -69.4 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

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
| 1 | 58.59 | -70.65 | 0.996 | 37.79 | -57.22 |
| 2 | 45.65 | -70.90 | 0.987 | 37.09 | -57.21 |
| 3 | 442.31 | -71.00 | 0.983 | 36.90 | -57.21 |

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
| 1 | 37.78 | -70.28 | 1.005 | 38.49 | -57.24 |
| 2 | 17.82 | -70.83 | 0.998 | 38.04 | -57.29 |
| 3 | 109.55 | -70.93 | 0.985 | 36.95 | -57.22 |
| 4 | 214.62 | -70.97 | 0.985 | 36.98 | -57.22 |
| 5 | 191.90 | -70.99 | 0.984 | 36.96 | -57.22 |
| 6 | 183.27 | -71.01 | 0.984 | 36.95 | -57.22 |
| 7 | 177.64 | -71.02 | 0.983 | 36.94 | -57.22 |
