# Closing speed on the stronger-calcium candidate

The phase factorial shows that opening-only acceleration lengthens the
first two intervals. Both-phase acceleration gives a closer first pair
but misses the human pause after the third spike. Test closing factor 0.5
instead of 1 on the gamma-0.002, removal-time-1000-ms candidate.

Keep Kv3 opening factor 0.5, source Kv3 density, soma Ca_LVA factor 0.5,
soma NaTg factor 1.10, sodium slope 5 mV, closure factor 0.15, and recovery
factor 1. Use the same phase build, mesh factor 9, and CVode tolerance
1e-10 at both calibration inputs. The current stronger-calcium traces
are the controls. Only Kv3 closing changes.

Test whether the absolute errors of the first two positive-spike intervals
both decrease against the human intervals at each input. Retain the third
interval and its error separately; do not call a first-pair improvement
a correct burst. Missing initial positive spikes invalidate this test.
Retain all events, first shape, and the preselected late pair after 1000 ms.
Report its absolute human error when the pair exists. Do not convert an
absent control pair into an improvement claim.

The combined parameter candidate is inferred. No promotion or circuit
qualification follows from this diagnostic. Do not read the reserved input.
