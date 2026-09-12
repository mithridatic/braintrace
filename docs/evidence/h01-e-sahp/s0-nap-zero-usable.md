# E-gain cell usable tier

Model: SP3 gain-split candidate s0-nap-zero (docs/specs/2026-09-07-h01-e-gain-split.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 200 pA | 1 | 0 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 pA | rate_hz |  | 0 | 0 |  |  | unavailable | no contract row | 15 percent of the human rate; no human repeats at this input |
| 200 pA | adaptation_ratio |  |  |  |  |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 200 pA | count |  | 1 | 0 | 0 |  | fail | fail | human repeat range x DLF over 5 repeats at this command |
| 200 pA | first_spike_ms |  | 206 |  | 82.6 |  | unavailable | no contract row | human repeat range x DLF over 5 repeats at this command |

## Retained per-cycle tables

### 200 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 207.76 | -70.00 | 0.967 | 36.22 | -54.72 |

### 200 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
