# L4-PYR cell usable tier

Model: Allen 527952884 perisomatic fit 626170709 source, unmodified (docs/specs/2026-09-07-h01-donor-allen-l4-import.md). Contract column: the approved
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
| 100 pA | width_ms | 2 | 0.721 | 0.968 | 0.144 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 3 | 0.733 | 0.968 | 0.147 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 4 | 0.749 | 0.967 | 0.15 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 5 | 0.754 | 0.967 | 0.151 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 6 | 0.757 | 0.966 | 0.151 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 7 | 0.76 | 0.966 | 0.152 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 8 | 0.773 | 0.966 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 9 | 0.764 | 0.965 | 0.153 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 10 | 0.761 | 0.965 | 0.152 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 11 | 0.776 | 0.965 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 12 | 0.775 | 0.965 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 13 | 0.767 | 0.965 | 0.153 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 14 | 0.784 | 0.964 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 15 | 0.783 | 0.964 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 16 | 0.779 | 0.964 | 0.156 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 17 | 0.783 | 0.964 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 18 | 0.779 | 0.964 | 0.156 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 19 | 0.796 | 0.964 | 0.159 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | ahp_mv | 1 | -69.8 | -70.3 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 2 | -69.2 | -70.9 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 3 | -70.1 | -70.9 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
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
| 100 pA | ahp_mv | 16 | -70.8 | -70.7 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 17 | -71 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 18 | -71 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 19 | -71 | -70.6 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | count |  | 20 | 19 | 4.41 |  | pass | fail | human repeat range x DLF over 4 repeats at this command |
| 100 pA | first_spike_ms |  | 32.4 | 33.9 | 6.23 |  | pass | no contract row | human repeat range x DLF over 4 repeats at this command |
| 90 pA | rate_hz |  | 12.6 | 7.69 | 1.89 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 90 pA | adaptation_ratio |  | 2.5 | 1.22 | 0.626 |  | fail | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 90 pA | width_ms | 1 | 0.703 | 0.973 | 0.141 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 2 | 0.753 | 0.968 | 0.151 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 3 | 0.776 | 0.967 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 4 | 0.774 | 0.967 | 0.155 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 5 | 0.787 | 0.966 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 6 | 0.788 | 0.966 | 0.158 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 7 | 0.786 | 0.966 | 0.157 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | width_ms | 8 | 0.797 | 0.966 | 0.159 | 0.0105 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 90 pA | ahp_mv | 1 | -69.8 | -70.4 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 2 | -69.4 | -71.5 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 3 | -69.3 | -70.9 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 4 | -69.7 | -70.9 | 2 | 1.06 | unresolvable | fail | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 5 | -69.9 | -70.9 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
| 90 pA | ahp_mv | 6 | -70 | -70.9 | 2 | 1.06 | unresolvable | pass | 2 mV of trough depth; spread from human repeats |
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
| 1 | 39.44 | -70.33 | 0.975 | 27.56 | -51.99 |
| 2 | 78.74 | -70.86 | 0.968 | 25.53 | -51.55 |
| 3 | 98.68 | -70.86 | 0.968 | 25.57 | -51.55 |
| 4 | 118.05 | -70.85 | 0.967 | 25.52 | -51.54 |
| 5 | 115.70 | -70.83 | 0.967 | 25.47 | -51.50 |
| 6 | 113.97 | -70.82 | 0.966 | 25.42 | -51.49 |
| 7 | 112.66 | -70.80 | 0.966 | 25.38 | -51.47 |
| 8 | 111.67 | -70.78 | 0.966 | 25.34 | -51.45 |
| 9 | 110.91 | -70.77 | 0.965 | 25.30 | -51.48 |
| 10 | 110.34 | -70.75 | 0.965 | 25.26 | -51.45 |
| 11 | 109.91 | -70.73 | 0.965 | 25.23 | -51.41 |
| 12 | 109.59 | -70.71 | 0.965 | 25.19 | -51.44 |
| 13 | 109.36 | -70.70 | 0.965 | 25.16 | -51.40 |
| 14 | 109.20 | -70.68 | 0.964 | 25.13 | -51.38 |
| 15 | 109.09 | -70.67 | 0.964 | 25.10 | -51.41 |
| 16 | 109.02 | -70.65 | 0.964 | 25.08 | -51.36 |
| 17 | 108.99 | -70.64 | 0.964 | 25.05 | -51.36 |
| 18 | 108.97 | -70.62 | 0.964 | 25.02 | -51.35 |
| 19 | 108.98 | -70.61 | 0.964 | 25.00 | -51.36 |

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
| 1 | 45.48 | -70.41 | 0.973 | 27.08 | -51.88 |
| 2 | 110.18 | -71.46 | 0.968 | 25.43 | -51.54 |
| 3 | 117.20 | -70.90 | 0.967 | 25.53 | -51.55 |
| 4 | 140.14 | -70.89 | 0.967 | 25.47 | -51.53 |
| 5 | 137.77 | -70.87 | 0.966 | 25.42 | -51.48 |
| 6 | 136.10 | -70.86 | 0.966 | 25.37 | -51.49 |
| 7 | 134.92 | -70.84 | 0.966 | 25.33 | -51.49 |
| 8 | 134.09 | -70.82 | 0.966 | 25.29 | -51.46 |
