# Adaptation on the opening-only candidate

The opening-only Kv3 candidate has a deeper first minimum than the both-fast
candidate, but still has too many events at low input. Test axonal calcium
removal time 1000 ms instead of the source 370.6410785 ms. Retain soma
NaTg factor 1.10, sodium closure factor 0.15, recovery factor 1, sodium slope
5 mV, soma Ca_LVA factor 0.5, Kv3 base time factor 0.5, and closing factor 1.
Keep all conductances and other parameters fixed. Use the same phase build,
mesh factor 9, and CVode tolerance 1e-10 at 0.19 and 0.27 nA.

The low-input control already exists. Run the high-input control and both
slow-removal cases. Compare every event, first shape, and interval. For
each input, test whether the interval between the first two positive spikes
whose upward crossings are at or after 1000 ms increases by at least 1 ms.
If fewer than two such events occur, this interval comparison is invalid;
report the absence separately rather than inventing a delay. Also report
whether the first upward crossing shifts by less than 0.1 ms.

These are diagnostic decisions, not human error tolerances. The input
regimes and late observation window are fixed before the runs. Do not
claim exclusive SK mediation: calcium also affects its reversal voltage.
Retain the reserved trace and do not promote a candidate from this test.
