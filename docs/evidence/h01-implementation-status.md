# H01 implementation status

The core two-cell simulation is built. The full validated system is not
complete. This audit separates missing capability from failed qualification.

| Capability | Current evidence | Remaining work |
| --- | --- | --- |
| H01 anatomy and annotations | Importers and selected measured E/I components are in use | Electrical region assignments remain inferred |
| Distinct E and I mechanisms | Frozen profiles, channel laws, calcium and cell builders exist. I transfer (SP2, 2026-09-07, amended gate: peak row 0.5 mV interpolated, timing gates unchanged): BrainCell at the copied x9 mesh and dt 0.005 reproduces NEURON fixed step at the same dt and mesh over 270-329.5 ms (rise 1.2e-9 ms, peak 5e-6 mV, width 2e-11 ms); both halving pairs inside the half gate. NEURON fixed step itself fails the CVode finalist on rise from event 6 (0.27 nA, 36 vs 37 events, drift to 7.4 ms) and event 3 (0.19 nA, drift to 14.8 ms); width and peak (-0.43 mV) inside their gates ([result](h01-i-transfer-result.md), [JSON](h01-i-transfer/sp2-i-decision.json)) | Full-train BrainCell arms untested: `a1-matched-027` killed at the 600 s watchdog twice (one mislaunch), `a1-matched-019/023` and `a0-maxcv-027` not launched; decision literal `untested`, B1 outcome selects the `integration` row (fixed step at dt 0.005 is not the CVode train); human-response qualification |
| Current stimulation and voltage recording | Donor and H01 runners save direct traces | Complete required input coverage |
| Synapses, delays and spike output | E/I circuit builder and four connection controls exist | Qualified default I response and circuit numerical checks |
| Direct E/I causal checks | Real H01 four-control audit records excitation and delayed E firing | Diagnostic I override is not a validated default |
| Compiled simulation | Cell.run uses brainstate.transform.for_loop | Runtime bottleneck has not been profiled |
| Source provenance | Reports separate anatomy, borrowed profiles and inferred contacts | Complete validation input/bias mapping |
| Human acceptance | Energetic search 2026-09-06 ([isolation](h01-isolation-decision.md), [multivari](h01-matryoshka2-decision.md)): I finalist reproduces the loop and trough at three inputs within repeatability limits and predicted 9 of 11 holdout rows, but the 0.23 nA holdout FAILS the approved contract on count and first peak ([I result](h01-i-energetic-result.md), [prediction](h01-prediction-i.md)); E gained an axonal initiation site and a calcium-SK dose under the usable-tier campaign (2026-09-07): 5 of 5 at 250 pA, 9 of 10 at 310 pA, every usable row but rate at 310 pA (by 0.1 Hz) ([E usable result](h01-e-usable-result.md)); sweep 53 is spent, sweep 55 sealed and unopened. | I: post-trough burst and accommodation, one evaluation in reserve; finalist stays experimental. E: one dose step and the sweep 55 prediction, two evaluations beyond the cap (returned) |
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
