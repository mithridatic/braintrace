# Sodium equilibrium-curve diagnostic

The combined candidate retains inward sodium current in its depolarized
nonspiking response. Recovery-speed changes do not change the stationary
gate value at fixed voltage. The whole-cell spatial check remains separate.

Before any new whole-cell calibration, inspect a change in the source
inactivation slope from 6 to 5 mV. Keep the midpoint at -56 mV, activation
parameters unchanged, conductance 0.5408169213181741 S/cm2, and ENa 50 mV.
These are diagnostic parameters, not measured human gate properties.

Use voltages -80, -70, -65, -56, -40, and -30 mV. Check the implemented
equilibrium gate against `1 / (1 + exp((V + 56) / slope))` independently.
At the removable pole -56 mV, the source perturbs its local voltage by
0.0001 mV. Report and allow the resulting difference up to 1e-5 in h;
do not mistake this source convention for a biological asymmetry.

Predeclare the directional outcome: below the midpoint, slope 5 must have
hInf greater than slope 6; above it, hInf must be smaller. At the midpoint,
both must agree with one half within 1e-5. Activation must stay unchanged.
Calculate and retain stationary sodium currents with the outward-positive
convention. Do not use this fixed-voltage check to claim a whole-cell rescue.
The source slope also enters its rate sum and therefore its time constant.
Record hTau as well: a raw slope intervention is not equilibrium-only.

## Whole-cell candidate

Set slopeh to 5 mV in soma and axon on the combined candidate. Keep soma
NaTg factor 1.10, closure factor 0.15, recovery factor 1, and all other
parameters unchanged. Use spatial factor 9 and CVode tolerance 1e-10 at
0.19 and 0.27 nA. The driver option must reject nonpositive or nonfinite
slopes, preserve the source value when omitted, and record the setting.

The late-firing rescue claim requires at least two complete positive-peak
events with upward crossings from 500 to 1270 ms at high input, where
the control has none. Report first-event errors and every subsequent event.
Also report sampled minima between the first consecutive events. Compare
their voltages and times with the human trace. Failure of the rescue does
not invalidate the fixed-voltage result. Success does not prove exclusive
mediation by equilibrium current because this slope also changes kinetics.
Do not read the reserved input or promote a candidate from this screen.

## Spatial check of the restored train

Repeat the slope-5 candidate at 0.27 nA on spatial factor 27. Keep the
same CVode tolerance and all physical parameters. Require at least two
complete positive-peak events after 500 ms on both meshes before calling
the rescue robust to this refinement. The slope-6 control on both meshes
has no such events. Retain paired event times, unmatched events, first
shape, and the first three post-spike minima. Report the time differences
even if the train survives. This is not full numerical convergence or
validation against the human trace.
