# E cell usable tier

This page scores the older energetic-search profile, which remains separate from
the unpromoted B3 continuation. The same frozen B3 parameters give 8/5 spikes at
250 pA, 10/10 at 310 pA, and 12/13 on the 350 pA prediction: see
[all B3 rows](h01-e-usable/b3-all-inputs-usable.md) and
[the completed continuation](h01-e-continuation-result.md).
The lower-current rate and higher-current cycle-2 width fail; no shared-profile
usable pass is established.

Model: kv3-ninety-ca133 candidate (energetic search, no E explanation passed). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 250 pA | 5 | 5 | pass |
| 310 pA | 10 | 6 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 pA | rate_hz |  | 4.79 | 4.45 | 0.718 |  | pass | no contract row | 15 percent of the human rate; no human repeats at this input |
| 250 pA | adaptation_ratio |  | 7.98 | 11.5 | 1.99 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 250 pA | width_ms | 1 | 0.936 | 0.963 | 0.187 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 2 | 1.13 | 0.958 | 0.227 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 3 | 0.975 | 0.952 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 4 | 0.971 | 0.951 | 0.194 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | width_ms | 5 | 0.963 | 0.951 | 0.193 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 250 pA | ahp_mv | 1 | -69.6 | -69.5 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 2 | -67.1 | -70 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 3 | -70.2 | -70.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 4 | -70.3 | -70.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 250 pA | ahp_mv | 5 | -70.6 | -70.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | rate_hz |  | 10 | 5.29 | 1.5 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 310 pA | adaptation_ratio |  | 9.95 | 13.8 | 2.49 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 310 pA | width_ms | 1 | 0.913 | 0.974 | 0.183 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 2 | 1.27 | 0.969 | 0.254 | 0.0118 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 3 | 1.09 | 0.952 | 0.219 | 0.0118 | pass | fail | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 4 | 0.986 | 0.951 | 0.197 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 5 | 0.988 | 0.951 | 0.198 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | width_ms | 6 | 0.977 | 0.951 | 0.195 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 310 pA | ahp_mv | 1 | -69.6 | -69.3 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 2 | -65 | -70.2 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 3 | -67.6 | -70.5 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 4 | -69.2 | -70.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 5 | -69.5 | -70.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 310 pA | ahp_mv | 6 | -69.6 | -70.1 | 2 | 1.01 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |

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
| 1 | 61.95 | -69.50 | 0.963 | 35.96 | -52.04 |
| 2 | 25.62 | -69.96 | 0.958 | 35.51 | -51.92 |
| 3 | 271.99 | -70.06 | 0.952 | 34.98 | -51.78 |
| 4 | 307.16 | -70.10 | 0.951 | 34.97 | -51.78 |
| 5 | 293.89 | -70.11 | 0.951 | 34.96 | -51.77 |

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
| 1 | 40.21 | -69.33 | 0.974 | 36.90 | -52.29 |
| 2 | 16.74 | -70.17 | 0.969 | 36.62 | -52.19 |
| 3 | 264.00 | -70.51 | 0.952 | 34.96 | -51.78 |
| 4 | 195.92 | -70.11 | 0.951 | 34.95 | -51.78 |
| 5 | 237.83 | -70.12 | 0.951 | 34.94 | -51.77 |
| 6 | 231.14 | -70.13 | 0.951 | 34.93 | -51.76 |
