# Subthreshold response persists under spatial refinement

Increasing the compartment count from 1239 to 3717 changes the 12
specified voltage samples by at most 0.000284433 mV. This passes the
0.01 mV numerical limit. Both runs retain all three response conditions:
late pulse decline, return below starting voltage, and no pulse spikes.

Every section count triples. Parent connections agree. Physical section
values pass the existing spatial audit tolerance, rtol=atol=1e-10.
The largest section-area difference is 1.136868e-13 um2. Applied genome,
passive parameters, total Ih conductance, input, and solver settings agree.
Both final recordings match their indexed raw samples. Input plateau
error is zero, excluding 1e-7 ms around command transitions. Both traces
are finite and end at 2100 ms.

The initial audit incorrectly required exact area equality. It stopped
before producing a result. Reusing the established geometry tolerance
resolves floating-point summation differences without changing the
voltage acceptance limit. Future spatial checks must use that geometry
rule and retain the measured differences.

The [full result](h01-l2-reversal-spatial-result.json) contains all direct
sample differences and source hashes. The result applies to these
subthreshold observations. Spiking numerical and human physiological
validation remain open.
