# Early intervals on the stronger-calcium candidate

The intervention changes Kv3 closing factor from 1 to 0.5 while retaining
opening factor 0.5, axonal gamma 0.002, calcium removal time 1000 ms, and
all other candidate settings. Both inputs use the same phase build, mesh
factor 9, and CVode tolerance 1e-10.

The predeclared first-pair improvement passes at both inputs. Every early
interval remains explicit below; this does not validate the full burst.

| Input, nA | Interval ordinal | Candidate, ms | Human, ms |
| --- | ---: | ---: | ---: |
| 0.19 | 1 | 9.035477 | 7.606420 |
| 0.19 | 2 | 9.245925 | 10.200971 |
| 0.19 | 3 | 11.925901 | 42.759876 |
| 0.27 | 1 | 4.970124 | 6.229948 |
| 0.27 | 2 | 4.223600 | 5.948006 |
| 0.27 | 3 | 4.047957 | 6.880304 |

At low input, the control's first pair was 21.597599 and 25.592240 ms.
At high input it was 9.140031 and 9.029490 ms. Faster closing improves
the absolute errors of both intervals at each input. The third low-input
interval becomes less like the human pause and remains much too short.

The selected late intervals are 157.811711 and 40.585695 ms, versus human
98.753820 and 25.259988 ms. The low-input control had no late pair; do not
claim a relative improvement from that undefined control interval.
Both candidate late intervals are too long even though early intervals
are short. Uniformly stronger late suppression cannot solve this pattern.

The candidate has 13 and 38 complete events, close to human totals 12
and 43 but with incorrect direct timing. First-onset errors remain
+7.647884 and -0.667983 ms. First-duration errors remain -0.052420 and
-0.046429 ms. No model is promoted.

The [audit](h01-pv-burst-adaptation-audit.json) retains every event, interval,
first error, and selected late pair. Run
`python -m docs.evidence.h01_pv_burst_adaptation_audit`. Missing initial
positive spikes invalidate the first-pair comparison. The reserved input
was not read. Spatial and physiological qualification remain open.
