# Sodium candidate passes the independent solver check

The fixed-step and CVode runs both contain five complete events with
matching peak signs. Maximum differences are 0.084196503 ms in onset,
0.002674163 mV in peak, and 0.000037849 ms in duration above -20 mV.
All corresponding events satisfy the unchanged numerical limits.

The [complete audit](h01-l2-sodium-recovery-solver-comparison.json)
retains every ordinal difference. Model identity, input setting, bias,
initial voltage, temperature, fitted genome, geometry, calcium factor,
sodium factor, and mechanism source and library hashes match. The
adaptive endpoint is 2100 ms. Its saved arrays exactly reproduce the
right-limit selection from the retained raw arrays. Input plateau error
is zero outside 1e-7 ms neighborhoods of source command transitions.

This check applies to sodium factor 2, calcium factor 1.5, and mesh factor
9. It does not establish candidate spatial convergence, agreement of the
entire continuous voltage trace, or human physiological validation.
The first-interval effect remains small and later timing errors remain.
Do not promote the candidate based on this numerical pass.
