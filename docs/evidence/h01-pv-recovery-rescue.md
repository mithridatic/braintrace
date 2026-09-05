# Recovery-only rescue test

The combined candidate retains soma NaTg factor 1.10 and closure factor
0.15. The intervention changes only recovery factor from 1 to 0.5.
Both inputs use mesh factor 9, CVode tolerance 1e-10, and source SK and
calcium parameters. The audit checks equality of the geometry and other
recorded model settings between each control and intervention.

The predeclared claim requires at least two complete positive-peak spikes
after 500 ms at 0.27 nA, where the control has none. The claim fails.

| Input, nA | Control complete events | Recovery complete events | Control late positive events | Recovery late positive events |
| --- | ---: | ---: | ---: | ---: |
| 0.19 | 39 | 2 | 27 | 0 |
| 0.27 | 4 | 1 | 0 | 0 |

Complete events cross -20 mV upward and then downward. They need not have
positive peaks. The reference peak detector reports only one positive spike
in each recovery intervention. The low-input control sustains positive
spikes through the pulse; the intervention does not.

At low input, the first onset changes by about 3.24e-7 ms, the first peak
by 3.94e-5 mV, and the first duration by 5.53e-6 ms. At high input, these
changes are about -1.51e-9 ms, 4.27e-5 mV, and 3.83e-6 ms. These small
values do not establish physical precision; they show that the first-event
response is nearly unchanged while the later response differs strongly.

Faster recovery does not rescue this candidate. Under the tested conditions,
it also removes sustained firing at low input. Do not generalize this to
all recovery changes or infer that a particular human channel is responsible.
The unique current pathway and numerical robustness of this transition
remain unresolved. No parameter is promoted.

The [audit](h01-pv-recovery-rescue-audit.json) preserves each event and interval.
Raw traces use the prefix `h01-pv-recovery-rescue-`. The reserved input was
not used. Reproduce the audit with `python -m docs.evidence.h01_pv_recovery_rescue`.
Missing events and nonpositive excursions cannot satisfy the rescue criterion.
The existing datum and reference tests passed: four tests.
