# E-gain cell usable tier

Model: SP3 gain-split candidate s1-ksahp-tail5 (docs/specs/2026-09-07-h01-e-gain-split.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 110 pA | 0 | 0 | pass |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 110 pA | rate_hz |  | 0 | 0 |  |  | unavailable | no contract row | 15 percent of the human rate; no human repeats at this input |
| 110 pA | adaptation_ratio |  |  |  |  |  | unavailable | no contract row | 25 percent of the human ratio; no human repeats at this input |

## Retained per-cycle tables

### 110 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |

### 110 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
