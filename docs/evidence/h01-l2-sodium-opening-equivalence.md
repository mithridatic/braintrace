# Default opening factor preserves the full recorded response

The new mechanism at opening factor one reproduces the frozen midpoint
control exactly. All 19 saved arrays match over 835322 samples, including
time, voltage, applied current, raw sample indices, and soma observations.

Metadata differ only in the explicit opening factor, mechanism source
and library hashes, and execution time. Only NaTs.mod differs among
the mechanism sources. All prepared source hashes match the manifest;
the compiled library matches the successful 36-case clamp result.

Raw mapping is exact. Recorded command plus bias has zero plateau error,
excluding 1e-7 ms around transitions. The finite trace ends at 2100 ms.
The [complete comparison](h01-l2-sodium-opening-equivalence.json) records
each array decision and artifact hash.

This passes the prerequisite for the factor-two opening intervention.
It does not establish physiological validity or numerical convergence
of the midpoint candidate. The same known waveform errors remain.
