# L4-PYR cell usable tier

Model: Allen 527952884 perisomatic fit 626170709 source-dt-half, unmodified (docs/specs/2026-09-07-h01-donor-allen-l4-import.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 100 pA | 20 | 19 | FAIL |
| 90 pA | 12 | 8 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 pA | rate_hz |  | 10.4 | 9.21 | 1.55 |  | pass | no contract row | 15 percent of the human rate; no human repeats at this input |
| 100 pA | adaptation_ratio |  | 2.02 | 1.38 | 0.505 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 100 pA | width_ms | 1 | 0.669 | 0.975 | 0.134 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 2 | 0.721 | 0.969 | 0.144 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 3 | 0.733 | 0.968 | 0.147 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 4 | 0.749 | 0.967 | 0.15 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 5 | 0.754 | 0.967 | 0.151 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 6 | 0.757 | 0.967 | 0.151 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 7 | 0.76 | 0.966 | 0.152 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 8 | 0.773 | 0.966 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 9 | 0.764 | 0.966 | 0.153 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 10 | 0.761 | 0.965 | 0.152 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 11 | 0.776 | 0.965 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 12 | 0.775 | 0.965 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 13 | 0.767 | 0.965 | 0.153 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 14 | 0.784 | 0.965 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 15 | 0.783 | 0.965 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 16 | 0.779 | 0.965 | 0.156 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 17 | 0.783 | 0.964 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 18 | 0.779 | 0.964 | 0.156 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 19 | 0.796 | 0.964 | 0.159 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | ahp_mv | 1 | -69.8 | -70.3 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 2 | -69.2 | -70.9 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 3 | -70.1 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 4 | -69.7 | -70.8 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 5 | -70.6 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 6 | -70.5 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 7 | -70.5 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 8 | -70.6 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 9 | -70.3 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 10 | -70.9 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 11 | -70.8 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 12 | -70.8 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 13 | -71.1 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 14 | -71.2 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 15 | -70.8 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 16 | -70.8 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 17 | -71 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 18 | -71 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 19 | -71 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | count |  | 20 | 19 | 4.41 |  | pass | fail | human repeat range x DLF over 4 repeats at this command |
| 100 pA | first_spike_ms |  | 32.4 | 33.9 | 6.23 |  | pass | no contract row | human repeat range x DLF over 4 repeats at this command |
| 90 pA | rate_hz |  | 12.6 | 7.69 | 1.89 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 90 pA | adaptation_ratio |  | 2.5 | 1.22 | 0.626 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 90 pA | width_ms | 1 | 0.703 | 0.974 | 0.141 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 2 | 0.753 | 0.968 | 0.151 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 3 | 0.776 | 0.968 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 4 | 0.774 | 0.967 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 5 | 0.787 | 0.967 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 6 | 0.788 | 0.966 | 0.158 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 7 | 0.786 | 0.966 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 8 | 0.797 | 0.966 | 0.159 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | ahp_mv | 1 | -69.8 | -70.4 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 2 | -69.4 | -71.5 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 3 | -69.3 | -70.9 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 4 | -69.7 | -70.9 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 5 | -69.9 | -70.9 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 6 | -70 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 7 | -69.7 | -70.8 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 8 | -70.3 | -70.8 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

### 100 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 34.56 | -69.84 | 0.669 | 27.34 | -55.69 |
| 2 | 50.28 | -69.19 | 0.721 | 26.69 | -54.41 |
| 3 | 72.20 | -70.06 | 0.733 | 26.44 | -54.03 |
| 4 | 70.10 | -69.72 | 0.749 | 25.72 | -52.38 |
| 5 | 87.68 | -70.63 | 0.754 | 25.84 | -52.16 |
| 6 | 82.68 | -70.47 | 0.757 | 25.84 | -52.53 |
| 7 | 90.32 | -70.47 | 0.760 | 26.03 | -52.72 |
| 8 | 108.16 | -70.56 | 0.773 | 25.94 | -51.59 |
| 9 | 88.80 | -70.25 | 0.764 | 25.53 | -52.47 |
| 10 | 89.98 | -70.91 | 0.761 | 25.47 | -52.22 |
| 11 | 110.28 | -70.84 | 0.776 | 25.06 | -52.47 |
| 12 | 100.18 | -70.75 | 0.775 | 25.19 | -51.88 |
| 13 | 99.16 | -71.13 | 0.767 | 24.88 | -52.72 |
| 14 | 110.28 | -71.19 | 0.784 | 25.22 | -51.16 |
| 15 | 105.86 | -70.84 | 0.783 | 25.69 | -51.69 |
| 16 | 110.46 | -70.78 | 0.779 | 25.44 | -52.06 |
| 17 | 123.84 | -70.97 | 0.783 | 25.28 | -51.03 |
| 18 | 109.08 | -71.00 | 0.779 | 25.50 | -52.34 |
| 19 | 124.00 | -71.03 | 0.796 | 24.97 | -50.66 |
| 20 | 101.64 | -70.78 | 0.786 | 25.66 | -51.78 |

### 100 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 39.44 | -70.32 | 0.975 | 27.55 | -52.00 |
| 2 | 78.81 | -70.87 | 0.969 | 25.52 | -51.56 |
| 3 | 98.72 | -70.84 | 0.968 | 25.56 | -51.56 |
| 4 | 118.11 | -70.83 | 0.967 | 25.50 | -51.53 |
| 5 | 115.76 | -70.82 | 0.967 | 25.46 | -51.53 |
| 6 | 114.02 | -70.80 | 0.967 | 25.41 | -51.51 |
| 7 | 112.72 | -70.79 | 0.966 | 25.37 | -51.49 |
| 8 | 111.72 | -70.77 | 0.966 | 25.33 | -51.49 |
| 9 | 110.96 | -70.75 | 0.966 | 25.29 | -51.47 |
| 10 | 110.39 | -70.74 | 0.965 | 25.25 | -51.45 |
| 11 | 109.95 | -70.72 | 0.965 | 25.22 | -51.44 |
| 12 | 109.63 | -70.70 | 0.965 | 25.18 | -51.44 |
| 13 | 109.41 | -70.68 | 0.965 | 25.15 | -51.42 |
| 14 | 109.24 | -70.67 | 0.965 | 25.12 | -51.42 |
| 15 | 109.13 | -70.65 | 0.965 | 25.09 | -51.40 |
| 16 | 109.06 | -70.64 | 0.965 | 25.06 | -51.39 |
| 17 | 109.03 | -70.62 | 0.964 | 25.04 | -51.39 |
| 18 | 109.02 | -70.61 | 0.964 | 25.01 | -51.37 |
| 19 | 109.03 | -70.59 | 0.964 | 24.99 | -51.36 |

### 90 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 35.98 | -69.75 | 0.703 | 29.78 | -55.06 |
| 2 | 41.90 | -69.38 | 0.753 | 29.44 | -53.28 |
| 3 | 59.08 | -69.28 | 0.776 | 28.75 | -52.06 |
| 4 | 64.78 | -69.69 | 0.774 | 28.78 | -52.59 |
| 5 | 80.42 | -69.91 | 0.787 | 28.31 | -51.94 |
| 6 | 82.00 | -69.97 | 0.788 | 28.06 | -51.41 |
| 7 | 81.18 | -69.72 | 0.786 | 28.31 | -52.44 |
| 8 | 92.56 | -70.25 | 0.797 | 27.91 | -51.97 |
| 9 | 82.36 | -69.72 | 0.791 | 28.00 | -52.41 |
| 10 | 85.88 | -70.06 | 0.795 | 28.00 | -52.16 |
| 11 | 97.12 | -70.38 | 0.812 | 27.50 | -50.84 |
| 12 | 104.84 | -70.38 | 0.807 | 27.50 | -51.19 |

### 90 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 45.48 | -70.40 | 0.974 | 27.07 | -51.90 |
| 2 | 110.28 | -71.46 | 0.968 | 25.42 | -51.54 |
| 3 | 117.23 | -70.88 | 0.968 | 25.51 | -51.55 |
| 4 | 140.20 | -70.87 | 0.967 | 25.46 | -51.54 |
| 5 | 137.84 | -70.86 | 0.967 | 25.41 | -51.52 |
| 6 | 136.16 | -70.84 | 0.966 | 25.36 | -51.49 |
| 7 | 134.98 | -70.82 | 0.966 | 25.32 | -51.47 |
| 8 | 134.15 | -70.80 | 0.966 | 25.28 | -51.46 |
