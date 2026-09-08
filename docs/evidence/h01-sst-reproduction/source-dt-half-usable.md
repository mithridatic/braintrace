# SST-L3 cell usable tier

Model: HL5MN1 published fit source-dt-half, unmodified (docs/specs/2026-09-07-h01-donor-hl5mn1-import.md). Contract column: the approved
1 mV / 0.05 ms / exact-count contract; rows it lacks say so. `unresolvable` means the
human spread exceeds half the usable limit.

## Counts (contract: exact)

| Input | Human | Model | Contract |
| --- | --- | --- | --- |
| 100 pA | 14 | 16 | FAIL |
| 150 pA | 34 | 30 | FAIL |

## Verdicts

| Input | Row | Cycle | Human | Model | Limit | Spread | Usable | Contract | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 pA | rate_hz |  | 13.4 | 16.1 | 2.02 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 100 pA | adaptation_ratio |  | 2.72 | 2.76 | 0.68 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 100 pA | width_ms | 1 | 0.383 | 0.577 | 0.0767 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 2 | 0.401 | 0.575 | 0.0801 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 3 | 0.409 | 0.573 | 0.0818 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 4 | 0.419 | 0.572 | 0.0838 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 5 | 0.423 | 0.572 | 0.0846 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 6 | 0.429 | 0.571 | 0.0858 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 7 | 0.428 | 0.57 | 0.0856 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 8 | 0.426 | 0.57 | 0.0853 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 9 | 0.429 | 0.57 | 0.0857 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 10 | 0.434 | 0.569 | 0.0869 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 11 | 0.433 | 0.569 | 0.0866 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 12 | 0.435 | 0.569 | 0.0869 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 13 | 0.434 | 0.569 | 0.0869 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 14 | 0.436 | 0.569 | 0.0873 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | ahp_mv | 1 | -74.4 | -73.5 | 2 | 0.827 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 2 | -73 | -74.4 | 2 | 0.827 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 3 | -72.9 | -74.6 | 2 | 0.827 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 4 | -72.5 | -74.7 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 5 | -72.5 | -74.8 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 6 | -71.7 | -74.8 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 7 | -71.8 | -74.8 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 8 | -72.1 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 9 | -72.1 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 10 | -71.8 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 11 | -71.7 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 12 | -71.8 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 13 | -72 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | ahp_mv | 14 | -71.4 | -74.9 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 100 pA | count |  | 14 | 16 | 2.94 |  | pass | fail | human repeat range x DLF over 4 repeats at this command |
| 100 pA | first_spike_ms |  | 24.3 | 27.1 | 1.59 |  | fail | no contract row | human repeat range x DLF over 4 repeats at this command |
| 150 pA | rate_hz |  | 34.6 | 30.6 | 5.19 |  | pass | no contract row | 15 percent of the human rate; no human repeats at this input |
| 150 pA | adaptation_ratio |  | 2.03 | 2.14 | 0.508 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 150 pA | width_ms | 1 | 0.381 | 0.579 | 0.0762 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 2 | 0.395 | 0.577 | 0.0791 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 3 | 0.408 | 0.574 | 0.0816 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 4 | 0.411 | 0.573 | 0.0823 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 5 | 0.421 | 0.572 | 0.0841 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 6 | 0.425 | 0.572 | 0.085 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 7 | 0.428 | 0.571 | 0.0856 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 8 | 0.432 | 0.571 | 0.0863 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 9 | 0.43 | 0.571 | 0.086 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 10 | 0.438 | 0.57 | 0.0876 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 11 | 0.435 | 0.57 | 0.087 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 12 | 0.438 | 0.57 | 0.0877 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 13 | 0.438 | 0.57 | 0.0876 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 14 | 0.439 | 0.57 | 0.0879 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 15 | 0.438 | 0.569 | 0.0875 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 16 | 0.438 | 0.569 | 0.0876 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 17 | 0.439 | 0.569 | 0.0879 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 18 | 0.444 | 0.569 | 0.0889 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 19 | 0.444 | 0.569 | 0.0888 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 20 | 0.445 | 0.569 | 0.089 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 21 | 0.447 | 0.569 | 0.0894 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 22 | 0.446 | 0.569 | 0.0892 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 23 | 0.446 | 0.568 | 0.0892 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 24 | 0.444 | 0.568 | 0.0889 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 25 | 0.449 | 0.568 | 0.0898 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 26 | 0.449 | 0.568 | 0.0899 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 27 | 0.448 | 0.568 | 0.0895 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 28 | 0.448 | 0.568 | 0.0896 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 29 | 0.45 | 0.568 | 0.0899 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 30 | 0.449 | 0.568 | 0.0898 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | ahp_mv | 1 | -75 | -72.7 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 2 | -73 | -73.5 | 2 | 0.827 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 3 | -72.8 | -74 | 2 | 0.827 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 4 | -71.9 | -74.1 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 5 | -71.9 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 6 | -71.5 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 7 | -71.5 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 8 | -71.5 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 9 | -71.3 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 10 | -71.1 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 11 | -71.1 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 12 | -70.9 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 13 | -71 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 14 | -70.9 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 15 | -71 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 16 | -71.1 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 17 | -71 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 18 | -70.2 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 19 | -70.2 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 20 | -70.6 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 21 | -70.5 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 22 | -70.1 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 23 | -70.6 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 24 | -70.4 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 25 | -70.3 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 26 | -70.5 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 27 | -70.5 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 28 | -70.1 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 29 | -70 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 30 | -70 | -74.4 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |

## Retained per-cycle tables

### 100 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 25.30 | -74.38 | 0.383 | 22.81 | -57.47 |
| 2 | 37.72 | -73.00 | 0.401 | 20.91 | -55.00 |
| 3 | 49.06 | -72.91 | 0.409 | 19.50 | -53.59 |
| 4 | 55.44 | -72.50 | 0.419 | 18.25 | -52.94 |
| 5 | 59.52 | -72.47 | 0.423 | 17.78 | -52.81 |
| 6 | 78.70 | -71.66 | 0.429 | 16.50 | -51.78 |
| 7 | 74.04 | -71.81 | 0.428 | 17.25 | -52.78 |
| 8 | 71.82 | -72.09 | 0.426 | 17.63 | -53.50 |
| 9 | 75.52 | -72.09 | 0.429 | 17.25 | -53.00 |
| 10 | 79.58 | -71.78 | 0.434 | 16.84 | -52.03 |
| 11 | 94.82 | -71.72 | 0.433 | 16.78 | -52.50 |
| 12 | 94.62 | -71.75 | 0.435 | 16.56 | -52.50 |
| 13 | 94.12 | -72.00 | 0.434 | 16.56 | -52.34 |
| 14 | 102.64 | -71.38 | 0.436 | 16.41 | -52.78 |

### 100 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 30.61 | -73.51 | 0.577 | 14.87 | -53.11 |
| 2 | 26.76 | -74.37 | 0.575 | 14.74 | -53.00 |
| 3 | 33.20 | -74.64 | 0.573 | 14.55 | -52.94 |
| 4 | 40.33 | -74.72 | 0.572 | 14.32 | -52.70 |
| 5 | 48.96 | -74.77 | 0.572 | 14.06 | -52.58 |
| 6 | 57.83 | -74.81 | 0.571 | 13.95 | -52.42 |
| 7 | 65.04 | -74.85 | 0.570 | 13.86 | -52.33 |
| 8 | 69.75 | -74.87 | 0.570 | 13.82 | -52.33 |
| 9 | 72.31 | -74.89 | 0.570 | 13.75 | -52.35 |
| 10 | 73.51 | -74.91 | 0.569 | 13.76 | -52.37 |
| 11 | 73.98 | -74.92 | 0.569 | 13.73 | -52.35 |
| 12 | 74.10 | -74.93 | 0.569 | 13.76 | -52.31 |
| 13 | 74.08 | -74.94 | 0.569 | 13.74 | -52.36 |
| 14 | 73.97 | -74.94 | 0.569 | 13.77 | -52.27 |
| 15 | 73.89 | -74.95 | 0.569 | 13.76 | -52.26 |
| 16 | 73.77 | -74.95 | 0.569 | 13.72 | -52.35 |

### 150 pA, human

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 14.08 | -74.97 | 0.381 | 22.69 | -57.81 |
| 2 | 17.00 | -73.00 | 0.395 | 20.75 | -56.47 |
| 3 | 20.92 | -72.75 | 0.408 | 19.25 | -54.34 |
| 4 | 22.72 | -71.88 | 0.411 | 18.12 | -53.84 |
| 5 | 23.10 | -71.91 | 0.421 | 17.44 | -54.03 |
| 6 | 23.98 | -71.47 | 0.425 | 16.88 | -53.16 |
| 7 | 26.24 | -71.47 | 0.428 | 16.28 | -53.28 |
| 8 | 26.66 | -71.47 | 0.432 | 16.06 | -52.69 |
| 9 | 26.66 | -71.28 | 0.430 | 15.84 | -52.00 |
| 10 | 27.42 | -71.06 | 0.438 | 15.50 | -52.84 |
| 11 | 27.58 | -71.09 | 0.435 | 15.00 | -52.91 |
| 12 | 29.26 | -70.94 | 0.438 | 14.66 | -51.59 |
| 13 | 28.02 | -70.97 | 0.438 | 15.16 | -52.66 |
| 14 | 29.58 | -70.91 | 0.439 | 14.59 | -51.81 |
| 15 | 28.12 | -70.97 | 0.438 | 15.00 | -53.47 |
| 16 | 29.30 | -71.06 | 0.438 | 14.31 | -52.25 |
| 17 | 30.00 | -70.97 | 0.439 | 14.41 | -52.66 |
| 18 | 28.92 | -70.19 | 0.444 | 14.22 | -51.94 |
| 19 | 29.94 | -70.22 | 0.444 | 13.88 | -51.28 |
| 20 | 30.32 | -70.59 | 0.445 | 14.16 | -52.31 |
| 21 | 28.90 | -70.47 | 0.447 | 14.38 | -51.72 |
| 22 | 30.10 | -70.09 | 0.446 | 13.72 | -51.16 |
| 23 | 29.74 | -70.59 | 0.446 | 14.50 | -52.66 |
| 24 | 31.42 | -70.41 | 0.444 | 14.00 | -52.00 |
| 25 | 30.84 | -70.28 | 0.449 | 14.00 | -51.22 |
| 26 | 32.62 | -70.47 | 0.449 | 14.00 | -51.22 |
| 27 | 31.88 | -70.53 | 0.448 | 14.00 | -52.38 |
| 28 | 31.94 | -70.06 | 0.448 | 13.75 | -51.53 |
| 29 | 34.78 | -69.97 | 0.450 | 13.50 | -51.50 |
| 30 | 31.52 | -70.03 | 0.449 | 14.03 | -51.34 |
| 31 | 32.94 | -70.28 | 0.452 | 13.81 | -51.63 |
| 32 | 33.44 | -70.16 | 0.448 | 14.09 | -51.81 |
| 33 | 33.24 | -69.91 | 0.451 | 13.63 | -51.38 |
| 34 | 34.52 | -70.25 | 0.451 | 13.72 | -51.81 |

### 150 pA, model

| Cycle | cycle_ms | ahp_mv | width_ms | peak_mv | threshold_mv |
| --- | --- | --- | --- | --- | --- |
| 1 | 19.40 | -72.69 | 0.579 | 15.51 | -53.49 |
| 2 | 16.64 | -73.52 | 0.577 | 15.40 | -53.36 |
| 3 | 20.51 | -73.99 | 0.574 | 14.96 | -53.05 |
| 4 | 24.45 | -74.15 | 0.573 | 14.63 | -52.79 |
| 5 | 27.29 | -74.20 | 0.572 | 14.38 | -52.58 |
| 6 | 29.33 | -74.22 | 0.572 | 14.23 | -52.47 |
| 7 | 30.81 | -74.24 | 0.571 | 14.11 | -52.47 |
| 8 | 31.90 | -74.25 | 0.571 | 14.02 | -52.44 |
| 9 | 32.71 | -74.27 | 0.571 | 14.02 | -52.36 |
| 10 | 33.30 | -74.28 | 0.570 | 13.98 | -52.31 |
| 11 | 33.76 | -74.30 | 0.570 | 13.90 | -52.35 |
| 12 | 34.11 | -74.31 | 0.570 | 13.93 | -52.29 |
| 13 | 34.40 | -74.32 | 0.570 | 13.90 | -52.23 |
| 14 | 34.60 | -74.33 | 0.570 | 13.89 | -52.22 |
| 15 | 34.79 | -74.34 | 0.569 | 13.87 | -52.21 |
| 16 | 34.93 | -74.35 | 0.569 | 13.88 | -52.24 |
| 17 | 35.04 | -74.36 | 0.569 | 13.83 | -52.29 |
| 18 | 35.14 | -74.36 | 0.569 | 13.82 | -52.29 |
| 19 | 35.21 | -74.37 | 0.569 | 13.85 | -52.20 |
| 20 | 35.28 | -74.38 | 0.569 | 13.83 | -52.27 |
| 21 | 35.34 | -74.38 | 0.569 | 13.83 | -52.27 |
| 22 | 35.38 | -74.39 | 0.569 | 13.85 | -52.23 |
| 23 | 35.43 | -74.39 | 0.568 | 13.85 | -52.21 |
| 24 | 35.45 | -74.39 | 0.568 | 13.79 | -52.28 |
| 25 | 35.49 | -74.40 | 0.568 | 13.81 | -52.26 |
| 26 | 35.51 | -74.40 | 0.568 | 13.83 | -52.24 |
| 27 | 35.52 | -74.41 | 0.568 | 13.82 | -52.18 |
| 28 | 35.55 | -74.41 | 0.568 | 13.82 | -52.18 |
| 29 | 35.56 | -74.41 | 0.568 | 13.84 | -52.21 |
| 30 | 35.57 | -74.41 | 0.568 | 13.80 | -52.26 |
