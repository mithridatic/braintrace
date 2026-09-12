# E-gain cell usable tier

Model: SP3 gain-split candidate s0-ksahp-a (docs/specs/2026-09-07-h01-e-gain-split.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 200 pA | 1 | 1 | pass |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 pA | rate_hz |  | 0 | 0 |  |  | unavailable | no contract row | 15 percent of the human rate; no human repeats at this input |
| 200 pA | adaptation_ratio |  |  |  |  |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 200 pA | width_ms | 1 | 0.967 | 0.987 | 0.193 | 0.0118 | pass | pass | 20 percent of this human cycle's width; spread from human repeats |
| 200 pA | ahp_mv | 1 | -70 | -71.1 | 2 | 1.01 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 200 pA | count |  | 1 | 1 | 0 |  | pass | pass | human repeat range x DLF over 5 repeats at this command |
| 200 pA | first_spike_ms |  | 206 | 127 | 82.6 |  | pass | no contract row | human repeat range x DLF over 5 repeats at this command |

## Retained per-cycle tables

### 200 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 207.76 | -70.00 | 0.967 | 36.22 | -54.72 |

### 200 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 130.61 | -71.07 | 0.987 | 37.16 | -57.21 |
