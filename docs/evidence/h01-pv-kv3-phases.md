# Kv3 opening and closing split

The source equations were extended in an isolated cache with a disabled-by-
default closing-time override. Sixteen fixed-voltage cases pass the analytic
exponential check for both gate directions. Source hashes are retained.

## Original comparison is invalid

The new build did not reproduce the old build's sample times exactly.
It recorded 113435 samples instead of 113512. Both runs have 47 events;
first rising times differ by about 2.25e-7 ms and last rising times by
about 3.47e-7 ms. These small event differences do not pass the predeclared
exact-array rule. The original attribution remains invalid. Do not claim
exact build equivalence. A regression preserves this failure boundary.

The [build comparison](h01-pv-kv3-phase-build-comparison.json) retains the
events. Exact sample-array identity is stricter than physical event agreement;
future designs must specify which form of equivalence they need in advance.

## Separate comparison within the new build

All four cases use explicit opening and closing factors in the same build
at 0.19 nA. Other parameters, geometry, and integration settings match.
The mixed-phase cases were retained; the two reference cases were run
under the separately specified within-build design.

| Phase accelerated | Advance at falling -65 mV, ms | First minimum, mV | Time to minimum from peak, ms | Complete events |
| --- | ---: | ---: | ---: | ---: |
| Neither | 0 | -74.664171 | 2.297774 | 34 |
| Both | +0.176383 | -73.328847 | 1.511688 | 47 |
| Opening only | +0.235089 | -77.634704 | 1.902270 | 33 |
| Closing only | -0.165211 | -69.184633 | 1.840123 | 56 |

Opening-only acceleration advances the return and makes the minimum more
negative. Closing-only acceleration delays the same falling crossing and
makes the minimum less negative. These are conditional effects in this model.
The narrower claim that opening-only reproduces the both-fast advance within
0.05 ms fails: the difference is 0.058706 ms. The threshold is not widened.

Opening-only first onset error is +7.647735 ms and duration error is
-0.052427 ms. Closing-only errors are +7.477408 and -0.010208 ms.
The candidate still does not match the human cell. Event counts do not
prove exclusive mediation of later firing; every event and interval remains
in the [factorial audit](h01-pv-kv3-phase-factorial.json).

Run `python -m docs.evidence.h01_pv_kv3_phase_audit --within-build` for this
comparison. The command without that flag still rejects the original
cross-build attribution. Spatial robustness and physiological validation
remain open. No parameters are promoted, and the reserved input was not read.
