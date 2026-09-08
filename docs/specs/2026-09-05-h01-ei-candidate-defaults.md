# Make the selected E/I candidates the H01 starting defaults

## User-directed scope

Focus on H01 E/I integration. Use the best available candidates from the
preceding session as defaults. Stop physiological tuning during this task.
Default selection does not confer physiological validation. The user has
explicitly authorized using incomplete candidates as starting defaults.

## Frozen choices

E: h01-l2-kv3-ninety-ca133, Allen specimen 541563728/model 626170538.
This is the latest joint candidate that retains five spikes, improved
peaks and recovery depth, and improved later intervals relative to its
Kv3-0.9 control. It is not better on every response than every prior model.
Use sodium opening factor 2, somatic sodium density factor 1.3, recovery
factor 1, Kv3 closing factor 0.9, calcium factor 1.33, uniform distributed
Ih factor 75, and passive reversal shift -4 mV. Copy all remaining
parameters from its pinned applied genome and physical metadata.

I: h01-pv-regional-mesh-axon2187, published putative-PV HL5BN1 candidate.
Use somatic NaTg density factor 1.1, sodium h closing factor 0.15,
recovery factor 1, h slope 5 mV, somatic Ca_LVA factor 0.5, somatic Kv3
opening/closing factors 0.5/0.5, axonal calcium gamma 0.004, and axonal
calcium decay 300 ms. Other parameters retain the pinned source fit.
The large NEURON segment count is validation evidence, not a required
BrainCell discretization setting.

## Implementation approach

1. Add immutable, versioned candidate profiles with exact source hashes,
   provenance, physical settings, and unresolved physiological errors.
   Keep published-source profiles available explicitly for reproduction.
2. Port the E candidate's full mechanisms into BrainCell. The existing
   H01 active builder uses a different two-channel Wilbers model; changing
   its two density defaults cannot reproduce the Allen candidate. Preserve
   the separate NaTs, calcium, SK, Ih, and other source laws, including
   state-dependent opening/closing changes and current sign conventions.
3. Extend the existing PV BrainCell mechanisms and builder to apply the
   selected I candidate's region-specific changes. Preserve source-mode
   behavior for independent comparisons.
4. Expose an H01 E/I builder with these profiles as defaults. Apply dynamics
   to measured H01 morphology through an explicit electrical-region map.
   Keep H01 identity and anatomy, borrowed channel laws, and inferred
   regions separate. An explicitly selected I profile must not silently
   relabel a source pyramidal cell as measured inhibitory anatomy.
   Reject ambiguous region assignments where they prevent applying a
   profile correctly; do not silently omit required axonal channels.
5. Add a runnable E/I example and source/candidate comparison evidence.
   Verify model construction, gate laws, initialized currents, and compiled
   responses. Use the two profile types for synaptic polarity in subsequent
   circuit integration; do not fabricate measured H01 partner identities.

No new physiological fitting, source downloads, or solver development
is needed merely to select defaults. Required implementation transfers
must preserve the selected candidates rather than substitute a simpler
model with the same name.

## Tests and completion gate

Use sibling tests and >90% coverage of changed production modules.
Test profile immutability, valid/invalid E/I choices, explicit overrides,
source preservation, region-specific assignments, conductance units,
gate singularities and bounds, calcium-current ownership, and recorded
provenance. Compare source/candidate channel and whole-cell responses
against the pinned NEURON references using declared numerical tolerances.
Use compiled BrainState/BrainCell execution, never Python time-step loops.

Complete this task only when actual H01 entry points select the profiles,
required mechanisms are implemented and exercised, and an executable E/I
example demonstrates the resulting cells. A profile registry alone is
not completion. Separate transfer-test failures from known human-response
mismatches. Do not claim the original full physiological/circuit goal is
complete from this default-integration milestone.

## Pinned reference metadata

- E metadata SHA256: 926efb3d5ae93b324bd653698d1c7d3fd478d9b94db1143b4eb933c5fcfa0f3a
- I metadata SHA256: d8d4d022457eb139a033ead73780f2eb12a0d58852503267912d961b1eb9d546
