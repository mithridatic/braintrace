# Qualify the numerical response of the calcium-1.33 candidate

Freeze h01-l2-kv3-ninety-ca133: calcium factor 1.33, Kv3 closing factor
0.9, sodium opening 2, sodium density 1.3, recovery 1, distributed Ih 75,
passive reversal shift -4 mV, source leak and remaining kinetics, sweep
50 plus recorded bias, mesh 9, CVode 1e-10, endpoint 2100 ms. Use the
same checked kv3-closing-source library.

Run two comparisons independently. Tighten tolerance to 1e-11 at mesh 9.
Separately run mesh 3 at tolerance 1e-10 and compare with saved mesh 9.
Do not change physical parameters or input. Require identical source
and library hashes and matching Kv3 metadata in both comparisons.

For each comparison require equal event counts and peak signs, every
onset change <=0.1 ms, every peak change <=0.1 mV, every duration above
-20 mV change <=0.01 ms, and every separate rising or falling phase
change <=0.01 ms. Preserve all individual differences and unmatched
events. Record each interval and recovery-minimum voltage and delay
difference descriptively; these do not add an extrema acceptance gate.

For the spatial comparison, each fine section must have three times
the coarse segment count and identical parent topology. Permit section
area roundoff within rtol=1e-10 and atol=1e-10 square micrometres. Check
regional areas and Ih distribution consistently. All remaining physical
metadata must match. Tolerance comparison permits only tolerance,
sample count, and execution time to differ. Spatial comparison permits
mesh factor, segment counts, sample count, time, and declared area
roundoff to differ.

Verify finite arrays, exact raw mappings, correct input plateaus within
1e-12 nA outside 1e-7 ms of command transitions, and exact 2100 ms
endpoint. Wrong setup or malformed data invalidates a comparison;
a valid numerical-limit failure rejects it. These tests do not establish
human physiological acceptance, subthreshold validation, or held-out
performance. Do not inspect the reserved response.
