# H01 verified-network throughput (SP1)

Generated 2026-09-07T15:24:48. dt 0.005 ms, 4 cells, 2 projections, 75,605 compartments. Cost qualification only; no physiology claim.

| Run | Solver | Duration ms | Steps | Construction s | Init+run s | Wall s | Peak RSS MB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d0p05 | h01_staggered_scan | 0.05 | 10 | 280.2 | 270.0 | 556.8 | 1350 | completed |
| d1 | h01_staggered_scan | 1.0 | 200 | 266.2 | 212.4 | 484.7 | 1342 | completed |
| d1r | h01_staggered_scan | 1.0 | 200 | 243.6 | 274.9 | 525.1 | 1331 | completed |
| d10 | h01_staggered_scan | 10.0 | 2000 | 246.4 | 363.2 | 615.1 | 1329 | completed |
| ctl-staggered-d1 | staggered | 1.0 | 200 | 239.3 | not tested | 844.1 | 8100 | killed |

## Fit T(steps) = a + b*steps (h01_staggered_scan, completed runs)

- a (init + compile): 245.39 s
- b (per step): 0.05764 s
- 200*b, seconds per simulated ms: 11.528 s
- residuals (s): 24.01, -44.54, 17.99, 2.53
- two-repeat decision limit on b: 0.97098 s/step; in seconds at the repeated point: 184.486
- prediction (0.05 ms within 156.96669389998715 s +/- limit): held
- three points linear within limit: held
- predicted 10 ms cost: 360.7 s (cap 900.0 s; allowed: True)

Killed or incomplete (untested, not negative): ctl-staggered-d1
