# PV channel transfer checks

The source is the released HL5BN1 model from ModelDB 267587.
Its pinned revision is `82cdd91bc93942ba19315371330a2412e064baf5`.
These channels include nonhuman kinetics fitted as part of a human cell model.
They are not measurements from H01.

## Independent rate reference

The original compiled NMODL equations provide the gate rate tables.
The instrumentation adds observation fields. It does not change equations.
See [the instrumentation record](h01-pv-rate-instrumentation.json) and
[the reference values](h01-pv-rate-oracle.json).

The reference driver calls `h.fcurrent()` after each voltage change.
This transfers section voltage to the mechanism before its rate procedure runs.
Without this call, the exported procedure used stale voltage.
Independent comparisons detected this error. The driver now checks that
voltage-dependent activation changes across the table.

## State and current checks

All ten mechanisms have stateful BrainCell channels.
The tests initialize gates at equilibrium and check zero initial derivatives.
They then change the clamped voltage, or calcium for SK.
A compiled BrainState loop integrates the channel derivatives with RK4.
The whole gate trace is compared with the exact exponential response from
the independently compiled rate table.
This RK4 driver is a test tool. It does not select the whole-cell solver.

The current check uses the source gate powers and reversal potential.
BrainCell current is positive inward.
SK reads calcium but carries potassium current.
Its current owner is potassium. It must not enter the calcium influx term.
Ih uses the source reversal potential of -45 mV.

The channel suite has 11 passing tests. The rate suite has 29 passing tests.
The rate tests also check removable poles, finite voltage derivatives,
positive time constants, and the SK calcium cutoff.

## Limits and next checks

The calcium state uses signed influx and exponential removal from CaDynamics.
Unit checks cover inward, zero, and outward current, with a dynamic Nernst voltage.
An assembled compartment records calcium, SK gates, and both ion currents.
Calcium entry raises concentration and SK activation in this check.
Each ion current matches only the channel that carries that ion.
The complete inhibitory cell remains to be built.
Check current reversal and ion ownership through the assembled cell as well.
Then compare numerical refinements against the accurate NEURON reference.
The human waveform mismatch in the released model remains unresolved.
Passing channel tests do not resolve it.

## Assembly corrections

The assembled test first failed because BrainCell builds a default ion instance.
The ion now defaults to the published soma removal time. The builder must
override that value in the axon.
The next failure showed that potassium-owned SK could read calcium before
the calcium state was allocated. The constructor now exposes resting calcium;
the standard initialization hook then allocates its state.
These cases remain covered by the assembled test.

Current probes use explicit ion instance names. Chemical symbols were not
accepted as aliases by this runtime. Keep names explicit in future builders.
Trace conversion stays inside the 64-bit context to avoid precision warnings.
