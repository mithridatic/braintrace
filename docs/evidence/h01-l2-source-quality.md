# Source metrics support retaining the repeated observations

All nine selected sweeps pass the available numeric checks: calibration
43 and 50, plus repeated trials 56--62. The
[audit](h01-l2-source-quality.json) retains source record IDs, file hash,
each reported value, and each decision. Source records were retrieved
from the Allen EphysSweep API for specimen 541563728.

The [Allen white paper, October 2017 v5, page 8](https://s3.amazonaws.com/webflow-prod-assets/689cfbd308fa7373b604d290/68ee796e6df1cc984c3ab432_Documentation_Cell_Types_Database-Electrophysiology_Overview.pdf)
specifies noise, recovery, bias, resistance, and manual-review checks.
Our audit applies its available numeric thresholds. The bridge/input
ratio cross-check uses the published resistance feature, not a verified
break-in resistance, and is labeled accordingly.

The selected public records do not expose final approval status. Manual
bridge review, electrode zeroing, final electrode drift, and the exact
relative-resistance gate remain unverified. Numeric passes do not prove
complete QC or stable cell health.

There is no failure in these measured checks that supports discarding
nonspiking sweeps 57 and 58. Their source spike counts are null; the raw
voltage traces independently establish absence of complete -20 mV events.
Repeated-response variability remains an observation, not an identified
noise mechanism or a tolerance for another input. No candidate parameters
or physiological acceptance limits were changed during this audit.
