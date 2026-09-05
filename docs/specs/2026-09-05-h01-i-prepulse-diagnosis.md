# H01 I pre-pulse state diagnosis

Continue the authorized H01 I response diagnosis. Retain the frozen candidate,
source component 678539249.0, electrical map, 1 nA pulse of 3 ms, solver,
initial voltage, and mesh. Change only pulse onset from 2 to 270 ms.
The longer wait matches the donor reference pre-pulse duration. It is not
asserted to be a measured H01 protocol or sufficient for equilibrium.

Use dt 0.005 ms and maximum CV length 10 um for the initial split. Save the
complete direct voltage trace and provenance. Compare against the existing
1 nA H01 I run with the same numerical settings. Check voltage immediately
before input, each threshold crossing, peak and time relative to pulse onset,
and post-pulse recovery. A positive peak alone does not establish a human-like
regenerative spike. If the outcome changes, refine the numerical settings
before treating the response as qualified. Do not change channel parameters
or lower the spike threshold to make the test pass.

Prediction under test: extending the pre-pulse period alone restores a
positive somatic spike for this input. A failed prediction excludes this
specific wait change as a sufficient correction; it does not exclude all
initial-state effects. Keep experiment details in evidence and only supported
conditional conclusions in the causal explanation.
