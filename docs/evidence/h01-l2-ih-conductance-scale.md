# Location and total Ih are distinct experimental factors

The current soma-only model has 0.205725683 nS of integrated maximum
Ih conductance. Spreading that same amount uniformly over soma, dend,
and apic gives density 1.332551813e-6 S/cm2. The selected areas are
592.059384, 5496.430399, and 9349.985571 um2, respectively.

Applying the published other-cell regional densities to these areas
would instead give 15.501834237 nS, a factor of 75.351964 above the
current total. This is hypothetical conductance accounting, not a model
run and not an estimate of this human cell's actual HCN abundance.
The [calculation](h01-l2-ih-conductance-scale.json) records areas, units,
source and distributed totals, and the comparison's explicit scope.
Density provenance is in the [source audit](h01-l2-human-ih-source.md).

The location split preserves the original small total. Its result must
not be used to accept or reject the much larger conductance represented
by the other model's densities. Conversely, adding those densities would
change both location and amount; it would not isolate either factor.
Neither maximum conductance nor its area integral fixes instantaneous
current, which also depends on local voltage and gating state.
