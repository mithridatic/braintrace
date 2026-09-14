# H01 explicit-spine assembly phase

Implemented on `feat/h01-braincell-biology` after milestone `2b419b4`.
This phase makes explicit spines usable through the real H01 builder and session
factory. Complete glia/myelin coupling and the all-104 learning gates remain open.

## Executable contract

`H01SpatialManifest` validates `h01-biology-spines-v1` against a SHA256 of the
complete H01 topology document. It requires the `h01-swc-um` frame, every active
instance's source hash, explicit spine dimensions/direction/provenance/origin,
contact-to-head assignments, and release probabilities keyed by contact ID.
Unknown fields, source/topology drift, missing/extra identities, invalid rates
and ambiguous head assignments are rejected before network construction.

The origin labels `measured`, `published_donor` and `synthetic` retain the
declaration supplied with the anatomy. A label alone does not certify a measured
reconstruction; source evidence still needs independent anatomical review.

`build_network(..., biology=manifest)` adds the declared necks and heads,
preserves source geometry and provenance, and remaps soma inputs/probes, output
locations, ordinary contacts and complete electrical regions. Named head
contacts are allowed only at a matching original attachment. Other contacts
remain on the corresponding source cable. Each added spine explicitly inherits
its parent's electrical family. No rates are fit or added parameters trained.
Contact IDs/order, magnitudes and optimizer layout are preserved.

`H01Session.build` accepts the same manifest in `settings['biology']`. It stores
its canonical JSON form, constructs the actual spines and keys release by active
contact identity. Rebuilding with the same topology/archive/settings followed
by `restore_physical` reproduces an interrupted wait, including physical state,
random release state and eligibility factors. A v2 checkpoint still requires an
already constructed matching session and whole-event boundaries. General
biological mutation and automatic reconstruction from the checkpoint alone
remain unsupported.

The saved [real-source topology](real-component-probe.topology.json) and
[biology manifest](real-component-probe.biology.json) are a concrete minimal
example. They contain one explicit synthetic spine and no invented connectivity.
Pass the topology to `H01Topology.from_dict`, load the verified `H01Archive`, and
pass the biological document to `build_network(..., biology=document)`; initialize
with `init_h01_network_states` before constructing the event model. Full sessions
instead receive the document through their numerical settings.

## Evidence

- **216 tests passed** in 161.68 s in the affected CPU gate; 17 warnings concern
  separate readout state and the existing approximate recurrence paths. This
  is not a full-repository or GPU gate. See `verification.xml`, `coverage.json`
  and `verification-summary.json` for exact scope and production hashes.
- A separate strengthened receptor test passed after that gate: a real queued
  conductance activated the mapped head receptor while the ordinary contact on
  the same cell stayed silent. The production implementation did not change.
- [Real source probe](real-component-probe.json): neuron `3178243558`, component
  0, has 2,748 source nodes and builds to 969 CVs with one synthetic spine.
  Construction took .184 s, initialization .384 s, and compilation plus one
  .1 ms event .957 s. All 160 cable steps at .000625 ms produced finite final
  voltages (approximately -83.987 to -83.974 mV). The entire probe had an external
  840-second timeout and finished well within it.
- This neuron is labelled L5 pyramidal, sparsely spiny. The probe uses the
  existing E default L2 donor dynamics, explicitly recorded as borrowed. It
  is one selected component out of 15, not the entire reconstructed neuron.
  Neither the spine nor this probe is a claim of matching human physiology.

## Coordinate and glial-placement audit

The [spatial frame audit](spatial-frame-audit.json) compares the same neuron ID
in the public c3 skeleton layer with all 15 original proofread fragments. It
uses the existing source conversions (SWC x/y/z: .032/.032/.033 um; c3: nm/1000),
without fitting offsets or transformations. Of 1,398 c3 vertices, 32.4% coincide
with a source vertex within 1 nm. Nearest-vertex distances have median .0716 um,
95th percentile .429 um and maximum 3.375 um. This supports coordinate-frame
compatibility at the checked cell, not branch correspondence or identical
segmentation across all structures.

The audit also measures nearest skeleton vertices between glial candidate
`63900941936` and all 104 explicitly selected neuronal components. The nearest
pair is .2013 um apart, involving neuron `620880207` and glial component **189**.
The next two candidates involve glial component 0. This is proximity evidence,
not a membrane-contact/ensheathment measurement. The closest pair is on a
fragment that would disappear if only the largest glial component were kept.
No glial components were discarded, joined or installed in a model.

Scripts `../probe_spine_assembly.py` and `../audit_spatial_frames.py` reproduce
these artifacts with local verified inputs. Raw skeleton NPZ files remain in
the ignored `.cache/h01-biology` directory; their identities, source URLs, units
and hashes are retained in the audit. The reference acquisition helper is shared
with the prior glial download; the reference skeleton here is a neuron.

## Edge cases and remaining tests

Covered cases include topology/contact order changes, stale SWC hashes, missing
and duplicate targets, negative/nonfinite probabilities, explicit origin,
source-soma preservation, inherited region coverage, ordinary-versus-head
delivery, and fresh-session restart with stochastic eligibility.

Fresh-session replay exposed tuple/list differences in serialized spine
directions. The session now retains the validated canonical manifest instead
of the caller's Python container representation. The exact replay assertions
were retained. A fixture also attempted duplicate directed contacts; it was
corrected to use distinct presynaptic cells, preserving the topology contract.

Next work: select and validate glial fragments and extracellular neighborhoods,
define species/membrane maps, connect real GABA/glutamate release, and assemble
glial K buffering and sheath electrical dynamics. Follow with measured spatial
and temporal refinement, closed-loop learning checks, and the all-104 gates.
The existing calcium runtime limit still applies; this fast spine-only cable
probe does not predict the cost of full coupled calcium simulation.
