# Layer-2 spatial refinement

Compare the retrieved source model with all section segment counts
multiplied by three. Use the command-only input and dt 0.000625 ms.
Keep the complete waveform, region lengths, areas, passive properties,
channel densities, initial-state rule, temperature, and solver fixed.
The two replacement axon sections must also be refined.

The source count is 1 + 2 * int(section_length_um / 40). Multiply this
count by a positive odd integer. Default factor 1 preserves the source.
Test zero, negative, even, boolean, and noninteger factors as invalid;
test section lengths on each side of the 40 um source-count boundary.

Require equal complete-event counts and peak-sign classes during the
main pulse. Every ordinal onset difference must be <=0.1 ms, every
sampled peak difference <=0.1 mV, and every duration-above-threshold
difference <=0.01 ms. Preserve all residuals and unmatched events.
These are spatial numerical limits, not physiological acceptance limits.

Require equal section names, parents, length, and integrated area. The
segment count must triple in every section. Missing or changed setup
invariants invalidate the comparison. Report floating-point geometry
differences using a relative and absolute tolerance of 1e-10.
If the declared event limits fail, retain the failure and diagnose it
before promoting a source fit or transferring it to H01 anatomy.

The legacy baseline metadata lacks parent fields. Use the separately
recorded source setup replay for those parent values, with an explicit
replay label. Do not rewrite the baseline artifact as if parents were
recorded during its simulation. All finer runs record parent fields.

The source-to-factor-three comparison fails the onset and peak limits,
while preserving seven complete events. Continue with factor three to
factor nine, retaining all declared limits and fixed model parameters.
This successive tripling must pass the same structural checks. The
calcium-removal intervention remains deferred until the mesh is selected.
