# Adaptive source reference

The corrected source-mesh CVode run reaches exactly 2100 ms and passes
the declared direct-event comparison against the fixed-step source run
at dt 0.000625 ms. Both have seven complete events with equal peak-sign
classes. Maximum onset difference is 0.079889642 ms, peak difference
0.003197983 mV, and duration difference 0.000040353 ms. All are within
the predeclared limits.

The [complete comparison](h01-l2-cvode-endpoint-comparison.json) records
setup invariants, input checks, endpoint, and raw-index verification.
Current plateaus match the source command exactly. The four input
transition times differ from the source clock by at most 1.55e-9 ms.
Integration takes 27.082944 seconds; no matched speedup is claimed.

Raw samples retain both sides of repeated event times. The derived
trace keeps the final sample only when voltage is exactly continuous.
Every derived sample has its raw index, and the index mapping is checked
exactly. Nine recording tests and thirteen driver guard tests pass.
The initial recorder and endpoint failures remain documented; they are
not recast as successful runs.

This qualifies source-mesh solver agreement. The factor-nine adaptive
comparison also passes: maximum onset change is 0.081720987 ms, peak
change 0.002615833 mV, and duration change 0.000037379 ms. Both runs
have seven events. Setup, raw-index mapping, input, and exact endpoint
checks pass. Integration takes 220.767072 seconds. The
[factor-nine record](h01-l2-cvode-space9-comparison.json) retains all
direct differences. Neither solver agreement nor spatial agreement
establishes human physiological validation. The doubled-removal calcium
candidate also passes its [own comparison](h01-l2-calcium-removal-solver-comparison.json),
with four events in each method and maximum onset difference 0.093174154 ms.
