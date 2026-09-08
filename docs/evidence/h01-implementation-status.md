# H01 implementation status

The core two-cell simulation is built. The full validated system is not
complete. This audit separates missing capability from failed qualification.

| Capability | Current evidence | Remaining work |
| --- | --- | --- |
| H01 anatomy and annotations | Importers and selected measured E/I components are in use | Electrical region assignments remain inferred |
| Distinct E and I mechanisms | Frozen profiles, channel laws, calcium and cell builders exist. I transfer (SP2, closed 2026-09-07 on the measured simulator identity, user decision): BrainCell at the copied x9 mesh and dt 0.005 reproduces NEURON at the same fixed step over the full 270-1270 ms train at 0.27 nA (36 = 36 events; rise 1.4e-7 ms, sampled peak 2.7e-6 mV, width 3.2e-9 ms, raw voltage 8.3e-5 mV; over 270-329.5 ms: 1.2e-9 ms, 5.4e-6 mV, 1.9e-11 ms). The remaining error is the fixed-step integrator against the CVode reference: first order in dt at both inputs (adjacent-dt ratios 1.97-2.00 over dt 0.005 -> 0.000625), the human 1 ms crossing tolerance met at dt 0.000625 at 0.27 nA (0.981 ms, 37/37 events) and at 1.864 ms at 0.19 nA at that dt ([result](h01-i-transfer-result.md), [JSON](h01-i-transfer/sp2-close-decision.json)) | dt requirement: a BrainCell or NEURON fixed-step run compared with a CVode reference must use dt <= 0.000625 ms at 0.27 nA and a finer step at 0.19 nA (one more halving projected, not measured); the 0.19 nA dt is unmeasured (six-run cap spent); the (c) literal 1e-6 mV / 1e-8 ms over the full train did not hold (float-level identity only); human-response qualification |
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

The 2026-09-05 reading of the full I transfer (42 events against 40, event 7 at -0.19 ms after halving dt) was made through unequal meshes; SP2 (2026-09-07) replaced it: with the NEURON per-branch counts copied, BrainCell and NEURON at the same fixed step are the same train to floating-point accumulation over 270-1270 ms, and the residual against the CVode finalist is the fixed step's first-order drift ([result](h01-i-transfer-result.md)). The standing directive "do not continue serial physiological tuning while this transfer failure remains" is retired. In its place: any BrainCell-versus-NEURON or fixed-step-versus-CVode comparison must copy the NEURON per-branch counts (`--mesh-from`) and use a step no coarser than 0.000625 ms at 0.27 nA (1 ms crossing tolerance met with 0.981 ms at event 37; at 0.19 nA the error is 1.864 ms at that step and the required step is unmeasured). A working simulation pipeline is available; combined human-response feasibility is not established.

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
