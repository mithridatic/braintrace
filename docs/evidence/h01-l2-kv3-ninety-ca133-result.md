# Faster calcium removal retains recovery depth and improves later intervals

The joint prediction passes. Five complete positive-peak spikes remain.
Each later absolute interval error decreases, the first interval changes
by only -0.012850 ms within its 0.1 ms diagnostic limit, and all four
minimum-voltage errors remain smaller than in the Kv3-factor-one reference.

| Interval | Human error (ms) | Absolute error reduction (ms) |
| --- | --- | --- |
| 1 | -8.324115 | -0.012850 |
| 2 | +51.566586 | +7.303481 |
| 3 | -0.223548 | +8.575495 |
| 4 | +20.554643 | +8.630014 |

Minimum-voltage errors remain +0.058685, -2.862961, +0.157327, and
+0.183755 mV. The final spike is still 64.475138 ms late. The first
interval is too short, the second and fourth are too long, and rising
phases remain too short. The passing joint prediction is not a full fit.

Only the somatic calcium-removal decay changes, from 679.276885 to
657.046005 ms, with its intervention record. Other physical metadata
and source/library hashes match. Raw mappings are exact, input plateau
errors are zero, all arrays are finite, and traces end at 2100 ms. The
[full result](h01-l2-kv3-ninety-ca133-result.json) retains all direct responses,
residuals, setup changes, and hashes.

This split supports a calcium-removal effect on later firing while
leaving recovery depth nearly unchanged at these settings. It does not
isolate SK mediation or establish a measured human decay constant.
Candidate-specific numerical, subthreshold, and reserved-input checks
remain open. Further increasing removal speed can worsen the already
near-target third interval, so uniform interval correction is not assured.
