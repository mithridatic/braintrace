# E cell usable tier

Model: b2-sk-half-ca-decay; exact frozen candidate, unpromoted. Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 310 pA | 10 | 9 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 310 pA | rate_hz |  | 10 | 8.39 | 1.5 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 310 pA | adaptation_ratio |  | 9.95 | 9.81 | 2.49 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 310 pA | width_ms | 1 | 0.913 | 1.01 | 0.183 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 2 | 1.27 | 1.01 | 0.254 | 0.0118 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 3 | 1.09 | 0.986 | 0.219 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 4 | 0.986 | 0.987 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 5 | 0.988 | 0.986 | 0.198 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 6 | 0.977 | 0.986 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 7 | 0.984 | 0.986 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 8 | 0.979 | 0.986 | 0.196 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 9 | 0.967 | 0.985 | 0.193 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | ahp_mv | 1 | -69.6 | -70 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 2 | -65 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 3 | -67.6 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 4 | -69.2 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 5 | -69.5 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 6 | -69.6 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 7 | -69.4 | -71.2 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 8 | -69.6 | -71.2 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 9 | -69.8 | -71.2 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

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
| 1 | 37.81 | -69.99 | 1.007 | 38.51 | -57.24 |
| 2 | 13.96 | -71.01 | 1.009 | 38.92 | -57.35 |
| 3 | 49.22 | -70.96 | 0.986 | 37.05 | -57.22 |
| 4 | 174.94 | -71.06 | 0.987 | 37.16 | -57.22 |
| 5 | 151.47 | -71.10 | 0.986 | 37.15 | -57.22 |
| 6 | 146.02 | -71.13 | 0.986 | 37.15 | -57.22 |
| 7 | 142.00 | -71.15 | 0.986 | 37.14 | -57.22 |
| 8 | 139.12 | -71.17 | 0.986 | 37.14 | -57.22 |
| 9 | 137.00 | -71.18 | 0.985 | 37.13 | -57.22 |
