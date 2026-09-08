# Ih location split with conserved maximum conductance

Use the bias-included sweep-43 midpoint with original leak and Ih rates.
Replace soma-only Ih with uniform density on soma, dend, and apic. Set
that density to source soma density times soma area divided by the sum
of those three areas. Use actual modeled membrane areas on mesh 9.
Keep axon passive. Verify integrated maximum conductance is conserved
to relative tolerance 1e-12. This is not constant instantaneous current:
local voltage and gate state remain free to respond.

Keep calcium factor 1.5, source sodium, initial voltage, temperature,
CVode 1e-10, full recorded-bias input, and 2100 ms endpoint fixed.
The uniform placement is inferred, motivated by another published human
model; its density and kinetics are not copied into this model.

Test sufficiency using the same direction conditions: voltage at 2019 ms
below 1120 ms, voltage at 2099 ms below 1019 ms, and no pulse spikes.
Retain all 12 direct voltage observations and their human residuals.
Failure rejects this location change as sufficient, not all dendritic
HCN mechanisms. Verify unique source target, positive finite regional
areas, source preservation, and total conductance before interpretation.
