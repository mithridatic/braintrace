# Check spatial sensitivity of the density candidate

Compare the saved mesh-9 density candidate against a new mesh-3 run.
Use the same sodium-activation-source library, sodium opening factor 2,
density factor 1.3, recovery factor 1, calcium factor 1.375, distributed
Ih factor 75, passive reversal shift -4 mV, original leak, sweep 50,
recorded bias, CVode tolerance 1e-10, and endpoint 2100 ms.

Each section must have exactly three times as many segments in mesh 9.
Require identical section topology and physical parameters. Allow section
area roundoff only within rtol=1e-10 and atol=1e-10 square micrometres.
Check regional areas and resulting Ih densities within the same relative
roundoff limit. Keep mechanism and compiled library hashes identical.

The new driver adds kv3_closing_factor=null when the option is omitted.
The old run lacks this metadata key. Account for this schema difference
only when the new value is null and both runs pin the identical original
Kv3 source and library. Do not treat a non-null factor as equivalent.

Require equal event counts and peak signs. Every onset change must be
<=0.1 ms, every peak change <=0.1 mV, every duration-above--20-mV change
<=0.01 ms, and each separate rising and falling phase change <=0.01 ms.
Retain each direct event and phase difference, including unmatched events.
A valid trace that fails a response limit rejects the spatial prediction.

Verify finite arrays, exact raw mapping, input plateau error <=1e-12 nA
outside 1e-7 ms around command edges, and endpoint 2100 ms. Incorrect
setup or malformed data invalidates the comparison. These are spatial
sensitivity limits for this candidate, not physiological acceptance.
Do not access the reserved response or use the unqualified Kv3 intervention.
