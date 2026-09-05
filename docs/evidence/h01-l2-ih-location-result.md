# Conserved-total Ih relocation is insufficient

The location sufficiency prediction is rejected. The distributed model
has no complete pulse spikes, but still rises between 1120 and 2019 ms
and remains above its 1019 ms voltage at 2099 ms.

| Time, ms | Human deflection, mV | Distributed model deflection, mV |
|---|---:|---:|
| 1019 | 0 | 0 |
| 1120 | 10.3125 | 12.097147 |
| 2019 | 8.8125 | 12.940414 |
| 2099 | -1.3125 | 1.436228 |

The [full audit](h01-l2-ih-location-result.json) retains all 12 voltage
samples and residuals. The source and redistributed maximum conductance
are 2.057256832e-10 S, equal within the specified relative tolerance of
1e-12. Regional density is 1.332551813e-6 S/cm2 on soma, dend, and apic.
Axon receives no Ih. Applied genome, geometry, source build, input,
initial state, and solver invariants pass. Core and extra raw mappings,
current plateau checks, and endpoint pass. Integration took 375.05 s;
this is a single measured duration, not a performance comparison.

The result rejects this distribution of the original small conductance
as sufficient. It does not test the much larger amount obtained by
applying another published cell's densities to this geometry. Those
amount and rate changes remain distinct factors. Nor does it prove that
HCN location is irrelevant in the human neuron. Candidate numerical
checks remain open. No candidate is promoted and no reserved input was
used.
