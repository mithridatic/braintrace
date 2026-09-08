# SST-L3 cell usable tier

Model: HL5MN1 published fit source, unmodified (docs/specs/2026-09-07-h01-donor-hl5mn1-import.md). Contract column: the approved
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
| 100 pA | rate_hz |  | 13.4 | 16 | 2.02 |  | fail | no contract row | 15 percent of the human rate; no human repeats at this input |
| 100 pA | adaptation_ratio |  | 2.72 | 2.77 | 0.68 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 100 pA | width_ms | 1 | 0.383 | 0.581 | 0.0767 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 2 | 0.401 | 0.578 | 0.0801 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 3 | 0.409 | 0.577 | 0.0818 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 4 | 0.419 | 0.576 | 0.0838 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 5 | 0.423 | 0.575 | 0.0846 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 6 | 0.429 | 0.574 | 0.0858 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 7 | 0.428 | 0.574 | 0.0856 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 8 | 0.426 | 0.573 | 0.0853 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 9 | 0.429 | 0.573 | 0.0857 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 10 | 0.434 | 0.573 | 0.0869 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 11 | 0.433 | 0.573 | 0.0866 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 12 | 0.435 | 0.572 | 0.0869 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 13 | 0.434 | 0.572 | 0.0869 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 100 pA | width_ms | 14 | 0.436 | 0.572 | 0.0873 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
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
| 150 pA | rate_hz |  | 34.6 | 30.5 | 5.19 |  | pass | no contract row | 15 percent of the human rate; no human repeats at this input |
| 150 pA | adaptation_ratio |  | 2.03 | 2.14 | 0.508 |  | pass | no contract row | 25 percent of the human ratio; no human repeats at this input |
| 150 pA | width_ms | 1 | 0.381 | 0.583 | 0.0762 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 2 | 0.395 | 0.579 | 0.0791 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 3 | 0.408 | 0.578 | 0.0816 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 4 | 0.411 | 0.576 | 0.0823 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 5 | 0.421 | 0.576 | 0.0841 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 6 | 0.425 | 0.575 | 0.085 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 7 | 0.428 | 0.575 | 0.0856 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 8 | 0.432 | 0.574 | 0.0863 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 9 | 0.43 | 0.574 | 0.086 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 10 | 0.438 | 0.574 | 0.0876 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 11 | 0.435 | 0.573 | 0.087 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 12 | 0.438 | 0.573 | 0.0877 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 13 | 0.438 | 0.573 | 0.0876 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 14 | 0.439 | 0.573 | 0.0879 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 15 | 0.438 | 0.573 | 0.0875 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 16 | 0.438 | 0.573 | 0.0876 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 17 | 0.439 | 0.572 | 0.0879 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 18 | 0.444 | 0.572 | 0.0889 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 19 | 0.444 | 0.572 | 0.0888 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 20 | 0.445 | 0.572 | 0.089 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 21 | 0.447 | 0.572 | 0.0894 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 22 | 0.446 | 0.572 | 0.0892 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 23 | 0.446 | 0.572 | 0.0892 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 24 | 0.444 | 0.572 | 0.0889 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 25 | 0.449 | 0.572 | 0.0898 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 26 | 0.449 | 0.572 | 0.0899 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 27 | 0.448 | 0.572 | 0.0895 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 28 | 0.448 | 0.572 | 0.0896 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 29 | 0.45 | 0.572 | 0.0899 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | width_ms | 30 | 0.449 | 0.572 | 0.0898 | 0.00259 | fail | fail | 20 percent of this human cycle's width; spread from human repeats |
| 150 pA | ahp_mv | 1 | -75 | -72.7 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 2 | -73 | -73.5 | 2 | 0.827 | pass | pass | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 3 | -72.8 | -74 | 2 | 0.827 | pass | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 4 | -71.9 | -74.1 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 5 | -71.9 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 6 | -71.5 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 7 | -71.5 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 8 | -71.5 | -74.2 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 9 | -71.3 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 10 | -71.1 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 11 | -71.1 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 12 | -70.9 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 13 | -71 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 14 | -70.9 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 15 | -71 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 16 | -71.1 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
| 150 pA | ahp_mv | 17 | -71 | -74.3 | 2 | 0.827 | fail | fail | 2 mV of trough depth; spread from human repeats |
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
| 1 | 30.67 | -73.49 | 0.581 | 14.79 | -52.96 |
| 2 | 26.77 | -74.36 | 0.578 | 14.58 | -52.98 |
| 3 | 33.35 | -74.63 | 0.577 | 14.44 | -52.72 |
| 4 | 40.62 | -74.70 | 0.576 | 14.04 | -52.67 |
| 5 | 49.47 | -74.75 | 0.575 | 13.75 | -52.47 |
| 6 | 58.45 | -74.80 | 0.574 | 13.81 | -52.21 |
| 7 | 65.70 | -74.83 | 0.574 | 13.71 | -52.13 |
| 8 | 70.35 | -74.86 | 0.573 | 13.66 | -52.09 |
| 9 | 72.88 | -74.88 | 0.573 | 13.66 | -52.10 |
| 10 | 74.03 | -74.90 | 0.573 | 13.65 | -52.10 |
| 11 | 74.45 | -74.91 | 0.573 | 13.43 | -52.24 |
| 12 | 74.55 | -74.92 | 0.572 | 13.62 | -52.15 |
| 13 | 74.53 | -74.93 | 0.572 | 13.59 | -52.17 |
| 14 | 74.43 | -74.93 | 0.572 | 13.49 | -52.26 |
| 15 | 74.33 | -74.94 | 0.572 | 13.50 | -52.26 |
| 16 | 74.23 | -74.94 | 0.572 | 13.46 | -52.23 |

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
| 1 | 19.45 | -72.68 | 0.583 | 15.31 | -53.27 |
| 2 | 16.67 | -73.51 | 0.579 | 15.31 | -53.29 |
| 3 | 20.60 | -73.98 | 0.578 | 14.85 | -52.91 |
| 4 | 24.57 | -74.14 | 0.576 | 14.49 | -52.64 |
| 5 | 27.45 | -74.19 | 0.576 | 14.17 | -52.53 |
| 6 | 29.45 | -74.21 | 0.575 | 13.95 | -52.47 |
| 7 | 30.95 | -74.23 | 0.575 | 13.93 | -52.33 |
| 8 | 32.02 | -74.24 | 0.574 | 13.79 | -52.30 |
| 9 | 32.82 | -74.26 | 0.574 | 13.72 | -52.27 |
| 10 | 33.43 | -74.27 | 0.574 | 13.78 | -52.21 |
| 11 | 33.88 | -74.28 | 0.573 | 13.80 | -52.16 |
| 12 | 34.20 | -74.30 | 0.573 | 13.79 | -52.08 |
| 13 | 34.50 | -74.31 | 0.573 | 13.58 | -52.22 |
| 14 | 34.70 | -74.32 | 0.573 | 13.67 | -52.25 |
| 15 | 34.85 | -74.33 | 0.573 | 13.65 | -52.24 |
| 16 | 35.03 | -74.34 | 0.573 | 13.72 | -52.11 |
| 17 | 35.13 | -74.34 | 0.572 | 13.70 | -52.12 |
| 18 | 35.20 | -74.35 | 0.572 | 13.73 | -52.05 |
| 19 | 35.30 | -74.36 | 0.572 | 13.69 | -52.00 |
| 20 | 35.35 | -74.36 | 0.572 | 13.69 | -52.10 |
| 21 | 35.40 | -74.37 | 0.572 | 13.66 | -51.98 |
| 22 | 35.48 | -74.38 | 0.572 | 13.71 | -52.08 |
| 23 | 35.50 | -74.38 | 0.572 | 13.71 | -52.05 |
| 24 | 35.53 | -74.38 | 0.572 | 13.65 | -52.11 |
| 25 | 35.55 | -74.39 | 0.572 | 13.59 | -52.20 |
| 26 | 35.58 | -74.39 | 0.572 | 13.71 | -52.02 |
| 27 | 35.60 | -74.40 | 0.572 | 13.70 | -52.06 |
| 28 | 35.63 | -74.40 | 0.572 | 13.70 | -52.03 |
| 29 | 35.63 | -74.40 | 0.572 | 13.53 | -52.15 |
| 30 | 35.65 | -74.40 | 0.572 | 13.51 | -52.15 |
