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
| Human acceptance | Energetic search 2026-09-06 ([isolation](h01-isolation-decision.md), [multivari](h01-matryoshka2-decision.md)): I finalist reproduces the loop and trough at three inputs and predicted the closed holdout ([I result](h01-i-energetic-result.md), [prediction](h01-prediction-i.md)); E rise rate two to three times the human's, load-limited ([E budgets](h01-e-energetic-stage-r-budgets.md)) | I: post-trough burst and accommodation, one evaluation in reserve. E: Stage X two-arm split (sodium vs dendritic load), then the sweep 53 prediction |
| I input datum | Command current only: the recorded 31.4 pA is a holding current the published fit absorbed ([forensics](h01-pv-bias-forensics.md)); bias-on residuals retired | None open |
| Measured connectivity | C3 edge list over all 104 cells: 123 candidate contacts, 3 endpoint-verified (incl. the I-to-E default), 6 cells joined ([edge list](h01-resolved-edge-list.json)) | E-to-I candidate 54906016 is not endpoint-supported; 5-voxel re-verification stalled and is untested; 71 candidates on one pair need a merge check |
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
