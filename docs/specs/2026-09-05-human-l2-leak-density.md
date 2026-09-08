# Passive leak amplitude and shape split

Multiply the four source g_pas values (soma, axon, dend, apic) by 1.5.
Keep their relative ratios, reversal potential, capacitance, morphology,
active channels, calcium factor 1.5, source Ih, and source sodium fixed.
Use the bias-included sweep-43 control, mesh 9, CVode 1e-10, full input
prehistory, and endpoint 2100 ms. This common multiplier is inferred.

The amplitude prediction requires smaller absolute deflection errors at
both 1120 and 2019 ms, compared with the unchanged control. Separately,
test full shape sufficiency with the existing two direction conditions
and absence of complete pulse spikes. Report the two decisions separately.
Retain all 12 voltage samples, baseline offset, and every human residual.
No amplitude pass can override a shape failure or qualify the cell.

Require exactly four unique target rows with positive finite densities.
Reject missing, duplicate, foreign-region, and invalid-factor inputs.
Verify that every non-target source value remains unchanged. Record
source fit hashes and the multiplier. Do not fit the reserved input.
