# Sodium activation speed is a separate diagnostic factor

The pinned NaTs source computes outward current as
`gbar * m^3 * h * (v - ena)`. Its activation state follows
`dm/dt = (mInf - m) / mTau`. The current preparation changes only
inactivation recovery, with its factor fixed at one in the current runs.

At fixed voltage, inactivation state, and sodium reversal, channel
density scales current. Activation time controls how quickly m approaches
its equilibrium. These interventions are distinct. A density experiment
does not establish what an activation-speed change will do.

At 34 degrees C, the source equations give mTau=0.217885 ms at -40 mV,
0.103465 ms at -20 mV, and 0.054833 ms at 0 mV. These are fixed-voltage
rate evaluations, not spike-phase durations. The
[source review](h01-l2-sodium-activation-source-review.json) retains hashes
and each rate evaluation, including the removable singularity at -40 mV.

A selective opening-time intervention could preserve equilibrium and
closing behavior while changing activation when mInf exceeds m. It would
require its own source isolation, fixed-voltage clamp, default-factor
equivalence, and whole-cell prediction tests. It has not been implemented
or tested here. The current waveform does not prove activation speed is
the cause, and these borrowed equations are not direct human measurements.
