# Adaptive recorder boundary

The first adaptive run fails the driver's strict-increase time check.
A second run captures raw samples before the same check and reproduces
the failure. It has 960661 samples, including 104999 duplicate-time
pairs and no negative time steps. Voltage is exactly equal across all
duplicate pairs. Applied current can jump across a pair by the stimulus
step. The [raw diagnosis](h01-l2-cvode-recorder-diagnosis.json) retains
the counts, exact voltage check, and example event times.

The failed assumption was that a variable-step recorder emits exactly
one sample at each time. Do not repair this by averaging current across
a discontinuity or dropping samples without retaining their provenance.
A proposed derived trace keeps the last sample at each duplicate time,
only after checking exact voltage continuity and no backward times.
The raw record must remain available with both sides of each event.
This conversion is not yet implemented or qualified.

The final recorded sample is at 2099.993687921267 ms, before the requested
2100 ms endpoint. The fixed-step endpoint assertion is also not directly
applicable to this recording. The complete main pulse through 2020 ms
is present, but endpoint handling needs an explicit adaptive-solver rule.
Integration took 31.258805 seconds, excluding output compression and
source setup. No matched runtime speedup is claimed.

This original failed trace is not promoted. A separate corrected run
uses the tested right-limit conversion and an explicit adaptive endpoint.
Its [completed comparison](h01-l2-adaptive-reference.md) records the
qualified scope. The factor-nine fixed-step run has also completed.
