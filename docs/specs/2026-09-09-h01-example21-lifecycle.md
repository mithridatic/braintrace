# H01 Example 21 computational lifecycle

Status: implementation authorized by the supplied plan; acceptance untested.

This specification supersedes the deferred-only scope and physiological training
prerequisites of `2026-09-08-h01-example21-adapter-contract.md`. Physiology is a
separate reported result, never inferred from computational integration.

## Population and compatibility

Use all 104 selected source components, pinned archive and annotation digests,
frozen donor parameters, and the existing sparse cable solver. Preserve source
points, radii, connectivity and annotation mappings. Correct the SWC reader's
relative-coordinate attachment comparison locally, with regression tests before
the fix. Import all components before constructing 4, 12, 40 and 104 cells.
Retain two constructible anatomical contacts and the blocked fragment contact.
Record construction, initialization, compilation, stepping, compartments and RSS
separately. No geometry simplification or surrogate dynamics is authorized.

## Model and learning

H01ArcModel implements update, step, interval, readout_features, readout and
reset_episode. Map 441 features to at most 32 distinct cells each using seeded
brainstate.random. Train registered ETP encoder weights and nonnegative contact
magnitudes; inject 0.35*tanh(drive) nA into somas. Use placed conductance receptors,
inherited polarity and 0.5 ms delay. Freeze geometry and donor physiology.
One ARC event holds current for 0.1 ms: 20 compiled 0.005 ms cable steps.
Read soma tanh((V_mV+65)/20) into a trainable 360-logit head. Preserve ARC losses,
decoding, scoring and grouped Muon. Eligibility must cover actual cable state;
unsupported compiler operations or excessive allocations are explicit blockers.
Reset all physical, event and eligibility state between episodes; padding advances
nothing. Repeated execution uses BrainState compiled transforms.

## Evolution and durability

Expose --model-backend h01 and a manifest on the evolution command. Reuse
EvolutionAdapter scheduling, corpus isolation, protected promotion and recovery.
Clone ranked cells with source/parent lineage and frozen physiology; copied
contacts are synthetic. Prune original or synthetic instances without deleting
historical provenance or allowing an empty network. Add ranked absent directed
contacts at the designated output (soma fallback) and target soma; mark assumptions.
Prune contacts with history retained. Identity-changing Dale arms are ineligible.
Remap optimizer moments by stable identity; zero new entries.

Use a separate versioned checkpoint with string source IDs, content-addressed
verified morphology, complete source/donor/instance/contact manifests, both clocks,
parameters, optimizer and coordinator identity. Reject cross-backend resume.
Persist episode boundaries and replay interruptions from the saved parent.

## Acceptance and costs

Require all 104 cells to import, construct, initialize and step finitely. Compare
the adapter against native Network.run including arrivals, delays, reset and
padding. Compare 0.005/0.0025 ms with equal event counts and 1 ms timing, 1 mV
voltage and 0.05 ms phase bounds in the measured window. Learning-rule checks use
finite-window chunked_online_param_gradients; require finite gradients, encoder
and active-contact updates and controlled descent. Execute causal controls and
every permitted mutation on real cells; verify reload and interruption recovery.
The final integration gate requires a real encoded ARC episode and pp-prop update
at 104 cells, followed by saved/restored continuation. Smaller runs do not pass it.

Co-locate *_test.py files; target >90% meaningful new-code coverage and affected
H01/compiler/oracle/Example21/lifecycle regressions. Run candidates sequentially,
reuse immutable construction assets and enforce explicit memory limits. Measure
costs before campaign estimates; runs estimated above 15 minutes require approval.
Report population, numerical agreement, learning, evolution, physiology and ARC
scores separately. No ARC score threshold or complete eight-round run is required.
