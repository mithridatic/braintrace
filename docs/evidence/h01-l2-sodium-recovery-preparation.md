# Selective sodium-recovery preparation

The isolated source copy changes only NaTs.mod. It adds a default-one
time factor when hInf > h. The source equilibrium, activation law, and
closing branch remain unchanged. All original mechanism files remain
intact. The [source manifest](h01-l2-sodium-recovery-source.json) records
each original and prepared digest. Two source-patch tests pass.

All 24 fixed-voltage cases pass the 1e-8 gate-error limit. They cover
four voltages, two factors, and recovery, closing, and equilibrium
initial states. The maximum h error is 6.729617e-13 and the maximum
activation-gate error is 6.061818e-14. All section voltages remain
exactly fixed. The [clamp record](h01-l2-sodium-recovery-clamp.json)
and saved trajectories retain every case.

These results verify the proposed gate law, not a human biological
mechanism. The factor-one full-cell run reproduces all 3,360,001 time,
voltage, and current samples of the saved calcium midpoint exactly.
The [control check](h01-l2-sodium-recovery-control-check.json) records
array equality and source hashes. This permits the factor-two test.
The driver records the sodium factor, mechanism source hashes, and
compiled-library hash. Fifteen CLI guard tests pass.

The sodium and calcium decision audits pass 29 tests with 100 percent
statement coverage of those two audit modules. Tests include rejected
timing and peak changes, changed input and mechanism sources, missing
events, and an empty human interval list. The empty list first exposed
an indexing error. The audit now reports an invalid reference. This
regression guards against assuming that a required observation exists.
A two-event result can pass the narrow first-interval prediction, but
the audit must retain the loss of later events.
