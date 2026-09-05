# Inhibitory transfer timing diagnosis

The completed full transfer has 42 events against 40 reference events.
The first rising-crossing failure is event 7. Diagnose this failure before
physiological fitting.

Use the existing donor reference runner with the frozen candidate, 0.27 nA
current, and maximum compartment length 2.5 um. Run to 330 ms with time
step 0.0003125 ms. Only the time step changes in the shared observation
window. Preserve the completed 0.000625 ms run as the control.

Compare every complete event from 270 to 330 ms. Require matching nonzero
counts, rising-crossing errors at most 0.1 ms, peak errors at most 0.1 mV,
and duration-above-minus-20-mV errors at most 0.01 ms. Retain failures and
each signed error. Do not align or shift traces. Check coverage, finite
samples, and events cut by a window boundary with the existing comparator.

The primary claim is: halving the time step is sufficient to bring event 7's
rising crossing within 0.1 ms of the reference at this mesh. The recorded
control error is -0.21473140577825234 ms. Accept this claim only if the
refined trace has the same complete event count in the window and event 7's
absolute crossing error is at most 0.1 ms. Otherwise reject it. An invalid
trace or changed input makes the test inconclusive, not a causal rejection.

The separate window-wide claim requires every event to pass every limit
above. Passing event 7 alone cannot establish that claim. Both decisions
remain pending; these statements were added while the refined run was live,
before its output was available. The original limits are unchanged.

A smaller error that still fails the limit does not establish sufficient
repair. A pass establishes sufficiency only for the stated intervention,
mesh, and observation. It does not identify a unique numerical mechanism,
exclude spatial error, or prove the full response. A failure does not by
itself prove a channel-law defect. No production change or human acceptance
decision follows from this diagnostic alone.
