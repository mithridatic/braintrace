# Calcium removal on the opening-only candidate

The intervention changes axonal calcium removal time from the source value
to 1000 ms. Both inputs use the same phase build, mesh factor 9, and CVode
tolerance 1e-10. Kv3 opening factor is 0.5 and closing factor is 1.
All other recorded model settings agree with their controls.

The selected interval is between the first two positive spikes whose upward
crossings occur at or after 1000 ms. This rule was set before the runs.

| Input, nA | Control interval, ms | Slow-removal interval, ms | Human interval, ms | First onset shift, ms |
| --- | ---: | ---: | ---: | ---: |
| 0.19 | 33.402142 | 38.694093 | 98.753820 | +0.000005447 |
| 0.27 | 12.305190 | 14.189611 | 25.259988 | +0.000000266 |

The interval-increase criterion of at least 1 ms passes at both inputs.
The first-onset shift criterion of less than 0.1 ms also passes. These
diagnostic decisions do not establish a human fit. First-onset errors
remain about +7.647740 and -0.667994 ms. First-spike duration errors remain
about -0.052427 and -0.046436 ms.

Low-input complete events change from 33 to 30. High-input events change
from 84 to 78. The recorded human traces have 12 and 43 events. Every event,
interval, and chosen late pair is retained in the audit; counts alone do
not qualify adaptation. The selected human pairs occur at 1094.255214 and
1193.009034 ms, and at 1018.158623 and 1043.418610 ms, respectively.

Axonal SK current agrees with conductance times gate times driving voltage
within 1e-10 mA/cm2 over each trace. Calcium, gate, current, and axonal
voltage at each soma rising crossing are retained. These checks do not
prove exclusive SK mediation: calcium also changes its reversal voltage.

Slower removal changes later behavior while leaving the first onset nearly
unchanged in this candidate. The late-interval mismatch remains substantial.
No model is promoted. Numerical and physiological qualification remain open.
The reserved human input was not read.

The [audit](h01-pv-opening-adaptation-audit.json) contains the full records.
Run `python -m docs.evidence.h01_pv_opening_adaptation_audit` to reproduce it.
Missing late spike pairs produce an invalid interval comparison; they are
not converted into an infinite interval or a claimed improvement.
