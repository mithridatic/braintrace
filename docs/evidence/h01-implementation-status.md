# H01 implementation status

The core two-cell simulation is built. The full validated system is not
complete. This audit separates missing capability from failed qualification.

| Capability | Current evidence | Remaining work |
| --- | --- | --- |
| H01 anatomy and annotations | Importers and selected measured E/I components are in use | Electrical region assignments remain inferred |
| Distinct E and I mechanisms | Frozen profiles, channel laws, calcium and cell builders exist | Full reference transfer and human-response qualification |
| Current stimulation and voltage recording | Donor and H01 runners save direct traces | Complete required input coverage |
| Synapses, delays and spike output | E/I circuit builder and four connection controls exist | Qualified default I response and circuit numerical checks |
| Direct E/I causal checks | Real H01 four-control audit records excitation and delayed E firing | Diagnostic I override is not a validated default |
| Compiled simulation | Cell.run uses brainstate.transform.for_loop | Runtime bottleneck has not been profiled |
| Source provenance | Reports separate anatomy, borrowed profiles and inferred contacts | Complete validation input/bias mapping |
| Human acceptance | Contract approved 2026-09-05; every record scored row by row ([scorecard](h01-contract-scorecard.md)); I and E bounded campaigns closed FAIL ([I](h01-i-campaign-result.md), [E](h01-e-campaign2-result.md)) | User decision on the named next families: I threshold (leak, reversal, Ih) under the recorded bias; E late subthreshold return |
| I input datum | Recorded 31.4 pA bias applied; zero-bias residuals retired ([datum](h01-pv-input-datum.json)) | Whether the published HL5BN1 fit applied this bias |
| Measured connectivity | One verified I-to-E contact; C3 relationship-index edge list over the 104 cells in progress ([pilot](h01-resolved-edge-list-pilot.json)) | Endpoint verification of the candidate E-to-I contact 54906016; per-cell partner counts |
| All 104 cells in a circuit | Current runnable circuit constructs one E and one I cell | Population-wide construction and qualification are not demonstrated |

The focused profile, cell, circuit and spike-output suite passed 40 tests in
55.67 seconds during this audit. It covers invalid inputs, source identity,
region constraints, propagating-spike masking, synaptic delivery and controls.
This run is not a whole-package coverage measurement or a physiological pass.

The full I transfer produced 42 events against 40 in the pinned reference.
Halving the time step at the same mesh does not repair event 7: its crossing
error is -0.192953 ms against the 0.1 ms limit. Both traces have eight complete
events in the diagnostic window. Event 8 also fails, with error -1.709768 ms.
The fixed sufficiency claim is false. See the
[direct audit](h01-pv-candidate-transfer-drift-halfdt-audit.json).

A working simulation pipeline is available, but numerical fidelity and
combined human-response feasibility are not established. Do not continue
serial physiological tuning while this transfer failure remains. Inspect
spatial discretization and implementation differences next. Do not assume
a new solver is required without isolating its limitation.

Current runners save traces after the compiled run and do not report progress
inside it. CPU activity does not identify the expensive operation or give a
reliable completion estimate. Separate construction, compilation, integration
and output before claiming a runtime cause.

The short runtime probe now separates a wait after the compiled loop from
later formatting. Explicit synchronization waits 6.85 seconds; subsequent
formatting calls stay below 0.1 seconds. All saved arrays remain exactly equal
to the control. This locates pending execution, not a costly kernel. Moving
the wait does not accelerate the model. See the
[synchronization evidence](h01-i-runtime-loop-sync-timing.json).
