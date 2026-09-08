# E-gain cell usable tier

Model: SP3 gain-split candidate g2-leak-150 (docs/specs/2026-09-07-h01-e-gain-split.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 110 pA | 0 | 0 | pass |
| 200 pA | 1 | 0 | FAIL |
| 250 pA | 5 | 0 | FAIL |
| 310 pA | 10 | 6 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 110 pA | rate_hz |  | 0 | 0 |  |  | unavailable | no contract row | 15 percent of the human rate; no human repeats at this input |
| 110 pA | adaptation_ratio |  |  |  |  |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 200 pA | rate_hz |  | 0 | 0 |  |  | unavailable | no contract row | 15 percent of the human rate; no human repeats at this input |
| 200 pA | adaptation_ratio |  |  |  |  |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 200 pA | count |  | 1 | 0 | 0 |  | fail | fail | human repeat range x DLF over 5 repeats at this command |
| 200 pA | first_spike_ms |  | 206 |  | 82.6 |  | unavailable | no contract row | human repeat range x DLF over 5 repeats at this command |
| 250 pA | rate_hz |  | 4.79 | 0 | 0.718 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 250 pA | adaptation_ratio |  | 7.98 |  | 1.99 |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 310 pA | rate_hz |  | 10 | 6.02 | 1.5 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 310 pA | adaptation_ratio |  | 9.95 | 5.19 | 2.49 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 310 pA | width_ms | 1 | 0.913 | 0.986 | 0.183 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 2 | 1.27 | 0.983 | 0.254 | 0.0118 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 3 | 1.09 | 0.98 | 0.219 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 4 | 0.986 | 0.979 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 5 | 0.988 | 0.979 | 0.198 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 6 | 0.977 | 0.979 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | ahp_mv | 1 | -69.6 | -70.3 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 2 | -65 | -70.6 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 3 | -67.6 | -70.8 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 4 | -69.2 | -70.9 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 5 | -69.5 | -70.9 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 6 | -69.6 | -71 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

### 110 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |

### 110 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |

### 200 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 207.76 | -70.00 | 0.967 | 36.22 | -54.72 |

### 200 pA, model

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
| 1 | 75.78 | -70.32 | 0.986 | 36.86 | -56.94 |
| 2 | 40.15 | -70.64 | 0.983 | 36.72 | -56.94 |
| 3 | 137.76 | -70.80 | 0.980 | 36.51 | -56.94 |
| 4 | 229.79 | -70.88 | 0.979 | 36.51 | -56.94 |
| 5 | 215.27 | -70.93 | 0.979 | 36.51 | -56.94 |
| 6 | 208.28 | -70.95 | 0.979 | 36.50 | -56.94 |
