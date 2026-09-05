# Gate speed and equilibrium current

The inspected diagnostic `NaTg.mod` uses:

`dh/dt = (hInf(V) - h) / tau(V, h)`.

For positive, finite tau, the stationary gate condition at fixed voltage
is `h = hInf(V)`. The closure and recovery factors change tau, not hInf.
The activation law is unchanged. Thus, these factors leave the fixed-voltage
stationary sodium current `gbar * mInf(V)^3 * hInf(V) * (V - ENa)` unchanged.
This is a deduction from the implemented equations. It does not require
a trend fit, and it does not imply unchanged transient or stability behavior.
In a coupled cell, changed gate rates can change whether a trajectory reaches
a fixed point. The observed late trace is not asserted to be an exact fixed point.

Direct interpolated soma observations at 1000 ms in the 0.27 nA runs are:

| Recovery factor | Voltage, mV | NaTg m | NaTg h | NaTg current, mA/cm2 outward positive |
| --- | ---: | ---: | ---: | ---: |
| 1 | -29.386425 | 0.793762 | 0.011311 | -0.242861 |
| 0.5 | -29.236306 | 0.795331 | 0.011425 | -0.246313 |
| 2 | -29.112345 | 0.797049 | 0.011250 | -0.243726 |

These observations come from the retained combined-candidate, recovery-rescue,
and recovery-slow NPZ traces. They are values at one time, not time averages.
The voltages differ, so this table is not a matched-voltage gate comparison.
The near equality of currents is not used to prove the equilibrium identity.
That identity follows directly from the gate equation above.

The sodium current is inward and nonzero in each nonspiking trace. Do not
explain the failure as complete closure of sodium channels. Do not infer
that sodium current alone maintains the voltage; potassium, calcium,
leak, Ih, axial flow, and applied current remain in the cell balance.
The [spatial check](h01-pv-failure-mesh.md) supports absence of late spikes
on two meshes; full numerical convergence remains unproven.

A subsequent intervention on the equilibrium gate curve would test a
different claim from an intervention on speed. Such a test must retain
first-spike shape, low-input onset, post-spike minima, and late events.
An arbitrary change to that curve is not a measured human parameter.
