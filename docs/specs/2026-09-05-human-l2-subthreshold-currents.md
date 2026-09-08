# Local subthreshold current diagnosis

Replay the bias-included sweep-43 midpoint with unchanged model, solver,
mesh, input, and endpoint. Add optional recordings at soma(0.5): each
mechanism's membrane current density, passive leak, capacitance current,
Ih and Im gates, and intracellular calcium. These are local observations,
not whole-cell currents. Use mA/cm2 for current density, mM for calcium,
and dimensionless gate values. Positive membrane current is outward.

Keep raw records and use exactly the core time record's right-limit
indices for all added arrays. Before interpreting currents, require exact
time, voltage, and applied-current equality with the existing uninstrumented
bias run. Preserve any failure. Check source mechanism pointers and units.

Report each current at the previously selected 12 observation times.
Identify interpolation. Differences within a trajectory can prioritize a
subsequent intervention but do not establish necessity, sufficiency, or
unique human causation. Do not infer whole-cell balance from one location.
Do not change conductances, initial states, or reserved-input status here.
