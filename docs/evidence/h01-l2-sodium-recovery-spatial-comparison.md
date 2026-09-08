# Sodium candidate passes one spatial refinement

Mesh factors 3 and 9 both produce five complete events. Every matched
event meets the unchanged limits. Maximum onset difference is
0.011144723 ms, peak difference is 0.011381885 mV, and duration above
-20 mV differs by at most 0.000080092 ms. Total segment counts increase
from 1239 to 3717. All section segment counts triple. Section geometry,
parent connections, fitted parameters, sodium recovery, calcium removal,
and mechanism source and library hashes remain consistent.

Time and applied-current arrays match exactly. The
[complete audit](h01-l2-sodium-recovery-spatial-comparison.json) retains
every ordinal difference. Both runs contain original parent records;
no legacy parent reconstruction was needed.

The audit originally omitted the newer intervention and mechanism-hash
fields. Four regression cases reproduced a false supported result when
one of those fields changed. The audit now rejects changed or one-sided
missing fields. Historical records with neither field remain usable for
their original scope. Twenty spatial-audit tests pass with 100 percent
statement coverage. Future intervention fields must enter the setup
invariants before the corresponding experiment is interpreted.

Together with the independent solver comparison, this result supports
numerical consistency for this candidate at the tested input. It is not
global convergence or human physiological validation. The full human
response still has material onset, waveform, and interval errors.
