# Independent adaptive-solver comparison

Keep the ongoing factor-nine fixed-step run intact. Independently compare
NEURON CVode at absolute tolerance 1e-10 against the saved source-mesh
fixed-step response at dt 0.000625 ms. Use the full command vector,
the source parameters, and a 2100 ms observation endpoint. The source
mesh remains factor 1 for this solver comparison.

Require identical model parameters, region geometry, and input waveform.
The integration method and sample times may differ. Check all four
current transitions and current plateaus on the source command clock.
Retain all complete pulse events and their direct differences.
Require equal counts and peak-sign classes, every onset difference
<=0.1 ms, every sampled peak difference <=0.1 mV, and every duration
above -20 mV difference <=0.01 ms. Missing required evidence invalidates
the comparison. This is solver agreement, not human validation.

Record integration wall time separately from source loading and output
compression. Do not claim a runtime speedup without equivalent measured
timing for both methods. A source-mesh pass does not qualify CVode on
the finer mesh or after a parameter intervention; check those cases
before relying on them.

The recorder diagnosis shows repeated times with exactly equal voltage
and potentially different applied current. Preserve the raw arrays.
A derived right-limit trace may retain the last sample at each repeated
time only if all arrays are finite, times never decrease, and voltage
is exactly equal across each duplicate pair. Retain raw sample indices.
Reject voltage jumps or malformed arrays instead of averaging them.
Use the recorded endpoint explicitly. The existing raw trace covers the
whole main pulse but does not establish exact arrival at 2100 ms.

The installed CVode API documents solve(tout) as returning states at
exactly tout and updating assigned variables. Use that one-shot call
for the adaptive branch after initialization. Retain continuerun for
fixed-step runs. Require the final recorded time to match the requested
endpoint within 1e-8 ms; do not fill an absent endpoint by extrapolation.

The source-mesh comparison passes with complete setup and input checks.
Next compare factor-nine CVode at the same tolerance against the completed
factor-nine fixed-step source run. Retain all event limits, the source
command, and the 2100 ms adaptive endpoint. This comparison must pass
before adaptive integration is used as a qualified factor-nine reference.
