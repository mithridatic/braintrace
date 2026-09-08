# Axonal calcium-entry scaling

The source optimizer fixes gamma_CaDynamics at 0.0005. This diagnostic
changes it to 0.002 in the axon only. The model retains removal time
1000 ms and the opening-only Kv3 candidate settings. Both inputs use
mesh factor 9 and CVode tolerance 1e-10. The override is inferred; it is
not a measured human calcium fraction.

At 0.19 nA, complete events decrease from 30 to 13. Only one positive
event occurs after 1000 ms, at 1134.189905 ms. The preselected late-interval
comparison is invalid because it requires two such events. Do not replace
the missing interval with infinity or choose a different window after the run.

At 0.27 nA, complete events decrease from 78 to 38. The selected interval
increases from 14.189611 to 40.003048 ms. The human target is 25.259988 ms.
The absolute error increases, so the predeclared improvement test is rejected.
The new selected pair rises at 1004.171139 and 1044.174186 ms.

First-onset shifts are only +0.000140 and +0.000009 ms. First-spike shape
errors remain essentially unchanged. The low-input first three intervals
are 21.597599, 25.592240, and 28.712857 ms, versus human 7.606420,
10.200971, and 42.759876 ms. High-input first intervals are 9.140031,
9.029490, and 9.690224 ms, versus human 6.229948, 5.948006, and 6.880304 ms.

Nearer total event counts do not establish the required direct response.
The candidate has a remaining early-burst error as well as a late-interval
error. Further late suppression alone cannot correct the early intervals.
No model is promoted. The reserved trace was not read.

The [audit](h01-pv-calcium-entry-audit.json) retains every event, interval,
and axonal calcium, SK gate, current, and voltage at each soma crossing.
SK current identities pass within 1e-10 mA/cm2. These checks do not prove
exclusive SK mediation of the timing changes. Run
`python -m docs.evidence.h01_pv_opening_adaptation_audit --entry`.
Forty-eight driver, datum, and crossing tests pass, including gamma range
checks and retention of the omitted source setting.
