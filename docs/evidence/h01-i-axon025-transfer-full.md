# Full inhibitory donor transfer with axon refinement

The full numerical transfer fails the fixed onset limit.

| Requirement | Observed | Result |
| --- | --- | --- |
| Same complete event count, 270–1270 ms | 40 reference; 40 BrainCell | Pass |
| Each onset error <= 0.1 ms | Maximum 1.165201799 ms; first failure at event 9 | Fail |
| Each peak error <= 0.1 mV | Maximum 0.068118477 mV | Pass |
| Each width error <= 0.01 ms | Maximum 0.000398242 ms | Pass |
| Exact shared prefix with the 330 ms run | Time, voltage, calcium and SK gate all equal | Pass |

The run uses 1439 CVs, axon maximum length 0.25 um and unchanged non-axon
intervals. The recorded runner time is 1712.886 seconds (28.55 minutes).
The original mesh produced 42 events against the same 40-event reference.
Axon refinement restores the count and preserves peak and width agreement,
but is insufficient for full spike-time transfer. It does not isolate the
cause of the remaining timing error. Do not promote this configuration as
qualified or relax the fixed limit.

This is a numerical donor-model comparison. The human recording has 43
events and remains a separate target. It is not a test on H01 anatomy.

The [audit](h01-i-axon025-transfer-full-audit.json) contains each event,
signed errors, shared-prefix checks, regional metadata and input hashes.
