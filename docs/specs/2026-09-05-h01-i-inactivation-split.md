# H01 I sodium inactivation split

Test the prediction that preventing transient sodium availability from falling
during the 2-to-5 ms pulse permits the soma voltage to cross 0 mV. Retain the
selected candidate, geometry, region map, initial state, 1 nA pulse, and mesh.
At each channel gate update during the pulse, replace the NaTg h equilibrium
with the current h only when the original equilibrium is lower. Keep recovery,
activation, conductance density, other channels, and behavior outside that
window unchanged. Apply this diagnostic clamp to all NaTg regions.

Record soma voltage and NaTg m and h. Require exact baseline voltage before
the intervention and nondecreasing soma h during the clamped window (within
1e-12). Retain all samples. Compare peak and each positive crossing. A positive
excursion under a nonphysical clamp is not a qualified action potential or a
new candidate. This tests an intervention in the model, not human causation.
Use dt 0.005 ms, then halve dt for any positive result. Do not alter defaults.

## Spatial dissection

Run soma-only and axon-only blocks on the same fixed map and time window.
Use the unique, frozen NaTg density of each region to select runtime channel
entries. Assert that soma and axon are the only NaTg regions and have distinct
densities before applying this selector. Record the density in the evidence;
it identifies existing entries and does not change their conductance.
Compare each response against the same unmodified baseline. Preserve the
exact pre-intervention voltage check. Require nondecreasing soma availability
only when soma is selected. The axon-only case must retain the source soma
closing law. These electrical regions remain inferred from sparse H01 labels.
