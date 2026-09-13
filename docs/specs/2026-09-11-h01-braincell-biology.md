# BRAINCELL biology in the H01 JAX execution path

Status: implementation in progress; tested mechanism and small-cable milestone.
Full spatial assembly and 104-neuron forward/training qualification are unfinished.
Current evidence and remaining gates: [biology report](../biology/README.md).

## Contract

Port the biological experiment families in bioRxiv
10.64898/2026.08.22.746419v3 to the existing JAX/BrainState runtime:
explicit spines, stochastic synaptic release, extracellular potassium and GABA,
astrocyte calcium reaction-diffusion and potassium buffering, and myelinated
axon potassium transport. Complete with the selected 104 H01 neurons and a
bounded matched ARC training comparison. Do not interpret feature presence,
reference agreement, or training as human physiological validation.

Reference: RusakovLab/BRAINCELL at
8acd1c23032a17af59ae1f813d142735fef8689c. Preserve per-file source digests and
license notices for reused source. Paper configurations and repository defaults
are separate: 500 synapses in the first results description conflicts with 316
spines in the methods; calcium buffer/diffusion settings also differ. Never
silently resolve a discrepancy into an exact-reproduction claim.

## Architecture

Reusable mechanisms live in `braintrace/biophysics`; H01-specific anatomy and
placement remain dataset adapters. Geometry and rates are immutable model
configuration. Continuous physical states belong to BrainState and use compiled
transforms for repeated evolution. Use brainstate.random for random draws.
Use sparse finite-volume transport with implicit differentiation, never dense
compartment-squared Jacobians. Explicit source-history diffusion is a separate
reference mode and must not be confused with conserved molecule injection.

Concentration units are mM, lengths um, time ms, voltage mV; finite-volume amount
is mM*um^3. Each membrane exchange is applied once with opposite signs to its
source/destination; clamped reservoirs and boundary exchange are recorded.
Reject nonpositive volumes, malformed topology, invalid rates, nonfinite state
and nonconverged solves. Do not clip invalid physical states into passing runs.

## Anatomy

Prefer verified matching H01 structures, then published human reconstructions,
then the paper's reconstructed source models, then source-constrained synthesis.
H01 c3 segmentation/skeletons, proofread meshes, astrocyte labels and myelin masks
are acquisition candidates, not certified completed structures. Verify identity,
frame, continuity, attachment and truncation. Preserve original neuron geometry;
store additions and inferred transformations separately, with stable IDs and
digests. No duplicated membrane area or contact. Default generation seed: 21.

## Integration

New mechanisms are opt-in. Keep existing H01 manifests and baseline equations.
Add a shared chemical environment to network stepping; close sparse pp-prop
influence over actual environmental coupling. Train only encoder, contact
magnitudes and readout; freeze rates, release probabilities and anatomy.
Stochastic derivatives condition on recorded masks, not on probability gradients.
Keep 441 input features, 360 logits and 0.1 ms event intervals; optional zero-input
physical intervals of 0, 1, 10 and 20 seconds precede answer requests in matched
arms. Never accelerate biological rates. Padding advances nothing.

Reset all physical, random, queue and eligibility state between independent
episodes. Versioned v2 checkpoints additionally support physical-step boundaries
inside long waits. Restore clocks, physical and chemical state, random counters,
eligibility, optimizer, queues and immutable manifests together. Legacy import
is explicit and only valid for the original mechanism set. Rebuild spatial maps
after topology mutation and preserve original/synthetic identity distinctions.

## Gates

1. Pin source evidence and test reusable units independently.
2. Validate spine/release and transport analytic limits, mass/boundary balance,
   receptor responses and recorded random-mask reproducibility.
3. Compare source-equivalent astrocyte and myelin mechanisms with isolated NEURON.
4. Integrate single systems, then 4/12/40/104-neuron populations.
5. Validate derivatives against AD/finite differences and learning through
   finite-window chunked_online_param_gradients. Keep approximate pp-prop
   qualification distinct from exact BPTT regimes.
6. Require restart equivalence, finite nonzero applicable updates, diagnostic
   descent, and the full 104-cell forward/learning gates.

Retain exact spike-count, 1 ms timing, 1 mV voltage and 0.05 ms phase limits.
Chemical peaks, decay and spatial profiles must change by <5% under refinement;
record full trajectories as well. New modules require meaningful sibling
`*_test.py` coverage above 90%. Write regression tests before bug fixes.

Local probes are capped at 15 minutes. Estimate measured time/memory before
requesting longer or paid runs. Stops remain incomplete. No GUI, assistant,
unrelated ARACHNE feature, unsourced channel-noise model or biological-rate fitting.

## Recorded implementation decisions, 2026-09-12

The calcium step solves reactions and diffusion together with matrix-free
Newton-Krylov and local 25-state block preconditioning. Operator splitting did
not meet the bounded source comparison. The selected internal step is .00125 ms
(four solves per .005 ms cable tick), based on the strong three-segment pulse
refinement fixture. This is not a qualification of long-duration waves or all
measured morphologies. Fixed ER and plasma-pump exchanges have explicit ledgers.

The initial v2 physical snapshot supports whole .1 ms event boundaries inside
waits. Checkpoints inside individual .005 ms cable substeps remain unfinished.
Restore requires an already constructed, manifest-matched session. Source v1
snapshots cannot silently omit new biology. Biological topology mutation is
rejected until a complete remapping adapter exists, including the case where
the contact count stays constant but contact identities change.

The first milestone accepted only fixed release probabilities in the H01
session factory. The subsequent spine assembly phase adds an explicit spatial
schema, but does not make the complete glia/myelin model available through
Example 21. Measured astrocyte acquisition must not be described as a completed
anatomical placement.

The finite-volume myelin transport is a distinct physical discretization. The
pinned RxD code and its comments disagree about axial diffusion and radial
geometry. Retain this distinction until independent reference experiments
resolve it; copied repository conversation notes are not scientific authority.

## Next phase: executable spine anatomy assembly

Add `h01-biology-spines-v1`, a canonical immutable manifest pinned to the complete
H01 topology document, source SWC digests and the `h01-swc-um` coordinate frame.
Every active instance and release contact must occur exactly once. Each spine
has a stable identity, explicit dimensions, source-coordinate attachment,
direction, provenance and measured/published-donor/synthetic origin. No anatomy
is generated or classified as measured implicitly.

The H01 builder will preserve original source objects, subdivide only instances
with explicit additions, inherit parent electrical regions, and remap soma,
output, ordinary synaptic sites and probes. A contact can move onto a named head
only when its original postsynaptic site matches that spine's attachment.
Contact/parameter identities and ordering stay fixed. Retain both original and
assembled placement evidence. Build and session paths must reject changed
topology/source hashes, unknown targets and unsupported schema fields before
expensive construction. v2 checkpoints bind the manifest; biological mutation
remains rejected until all spatial features can be transported coherently.

Gate this phase on meaningful malformed-manifest tests, geometry/region/contact
preservation, an initialized compiled cable step, and an actual H01 session
save/rebuild/restore replay with explicit spines and stochastic release. This
phase does not promote the fragmented astrocyte candidate, generate unverified
myelin, or claim full 104-neuron biological qualification.

## Neuron-glia coupling phase

Own neuronal and glial intracellular K pools in one shared environment, with
disjoint, complete membrane bindings. Implement the pinned Kir4.1 law as a real
BrainCell potassium channel, so the glial cable voltage and its axial current
evolve rather than treating the astrocyte as an unlabelled voltage clamp.
The initial supported glial electrical model is explicitly Kir-only; no claim
of a complete astrocyte channel repertoire or electrogenic calcium feedback.

Read the network's actual end-of-step population spikes, using separately
declared inhibitory GABA and excitatory glutamate source maps. GABA sites retain
their source settings; glutamate site dose and transport must be explicitly
supplied, since the source calcium experiment's bath concentration does not
determine a neuronal vesicle dose. This bridge is a declared modeling addition.
Account for molecule release, uptake and bath exchange. Sample neuronal/glial K
currents before either advances, apply them exactly once after cable dynamics,
then advance transmitter diffusion and mapped astrocyte calcium. New chemistry
affects the next neuronal cable tick. Freeze all geometry and biological rates.

The coupled driver must own glial cable, chemical, calcium and RNG states for
checkpoint/reset; the neuronal population remains the ARC input/output domain.
Require explicit basis metadata, source ownership, consistent clocks, valid
pool coverage and exact extracellular volume agreement. Reject overlapping
bindings and ambiguous or missing source/astrocyte maps before compilation.

Gate on actual evoked neuronal release, glial K uptake with opposite pool debit,
neuronal receptor feedback, nonzero glutamate-to-calcium response, conservation,
compiled gradients where mathematically applicable, reset and fresh-model
checkpoint replay. Run bounded small coupled systems first, within 15 minutes.
Measured glial-fragment selection, full coupled H01 factory construction,
myelin electrophysiology and all-104 training remain separate qualification work.

Phase result (2026-09-12): the small-system coupling gate passed, with 185
affected CPU tests and 100% statement coverage of the three new production
modules. Numerical probe, independent Kir cable reference and qualification
boundaries are recorded in `docs/biology/neuroglial-phase/README.md`. The six
existing session readout-state warnings remain visible. This completes the
bounded coupling phase, not the complete biological H01 implementation.

## Measured glial fragment assembly phase

Introduce an immutable source-pinned glial fragment manifest. Select a complete
connected component by an explicit original vertex anchor, never by proximity
alone or by silently joining fragments. Pin the source artifact, identity,
physical coordinate frame, selection rationale and maximum selected size.
Validate finite positive radii, nonzero edge lengths, valid unique undirected
edges and a forest topology. Preserve all source coordinates and tapered radii
under the established nm-to-um conversion. Record original vertex and edge
identities and an inventory of every excluded component.

Construct a BrainCell morphology from complete source paths between junctions;
retain every source segment and do not create soma geometry for a fragment.
Build a Kir-only dynamic cable with explicitly declared electrical settings.
Pin reconstructed geometry and settings for deterministic rebuild, while keeping
glia outside the neuronal ARC domain. Source-skeleton anatomy is measured;
electrical rates and sealed fragment boundaries are modeling assumptions.

Test source tampering, malformed geometry, cycles/duplicates, disconnected
selection, size limits, preservation of taper/coordinates/edge identities,
compiled cable evolution and deterministic fresh reconstruction. Execute a
bounded probe on the previously acquired candidate's explicitly chosen local
fragment, retaining all exclusions and the prior proximity-audit identity.
This phase does not infer synapses or extracellular neighborhoods, approximate
tapered calcium volumes as cylinders, or qualify a complete astrocyte/session.

Phase result (2026-09-12): 97 affected CPU tests and a separate axial-current
test passed; both new modules reached 100% statement coverage. Source astrocyte
63900941936's selected 571-vertex component was preserved in a 148-CV cable,
with 256 other components explicitly excluded. A 0.1 ms fixed-pool trajectory
matched fresh reconstruction exactly. Evidence and unresolved tapered-calcium,
extracellular mapping and session integration are in
`docs/biology/glia-assembly-phase/README.md`.

## Tapered cable chemical geometry phase

Map an initialized cable's exact CV ordering and source branch intervals into
chemical volumes. Integrate each clipped linear frustum's volume; preserve
BrainCell's membrane area and half-CV axial geometric resistance. Require a
complete nonoverlapping branch partition and agreement with the electrical
geometry. Do not substitute midpoint or mean-radius cylinders.

Use the pinned four-shell volume fractions for homothetic tapered annuli.
Integrate source radial conductance per unit cable length. At every shared
zero-volume boundary, eliminate the common concentration with Kirchhoff flux
balance: pairwise conductance is g_i*g_j/sum(g). This retains sibling diffusion
at junctions rather than approximating a fork as independent parent-child links.
Transport stays conservative; closed endpoints introduce no reservoir volume.
This is a declared tapered finite-volume extension, not verified source RxD or
three-dimensional taper equivalence.

Allow AstrocyteCalcium to use this geometry, with exact shell volumes and actual
membrane-area/outer-shell-volume ratio. Reuse the same full CV volumes for K
pool mapping. Preserve the existing cylindrical constructor and its independent
reference gates. Pin the CV geometry identity and support only source cable
geometries whose physical interpretation can be verified.

Gate on analytic tapered volume/resistance, cylinder-limit agreement, fork
Schur-complement flux, closed-system conservation, calcium reaction/buffer mass
balance, continuous derivatives and source subdivision invariance. Execute a
bounded 148-CV measured-fragment calcium/K probe with explicit diagnostic
extracellular pools; record exact reset/reconstruction replay and all assumptions.
No extracellular anatomy or neuronal release sites are inferred from proximity.

Phase result (2026-09-12): 201 affected CPU tests passed, with 99.05% new geometry
module coverage and 100% updated calcium-module coverage. The measured 148-CV
probe used identical full-CV K and calcium shell volumes, passed conservative
K and ER/pump-accounted calcium ledgers, and exactly replayed all physical states
after fresh snapshot restoration and reset. The 0.015 ms diagnostic and explicit
taper/extracellular assumptions are recorded in
`docs/biology/tapered-chemistry-phase/README.md`. Full session assembly remains next.

## Explicit neuroglial session assembly phase

Add h01-biology-neuroglia-v1 with nested topology-pinned spine declarations,
one explicitly selected glial fragment and electrical settings, exact CV geometry
digests and membrane-to-volume indices, extracellular centers/accessible volumes
and species-specific transport/clearance/boundary rates. Require origin/basis
declarations and explicit neuronal release destinations, doses, seeds and E/I
source identities. No nearest-neighbor assignment or release-site inference.

Construct neurons with environment K and tonic GABA before initialization; keep
the glia outside ARC populations. Reconstruct all physical volumes and reject
stale geometry maps before attaching chemistry. Bind the complete driver before
learner compilation, with the same fixed physical clock. Resolve glial source
assets by digest through caller-supplied local paths; paths are not biological
identity and missing assets must fail explicitly.

Canonical session biology must pin physical snapshots automatically and reject
conflicting supplied manifests. Preserve legacy release/spine behavior and the
existing biological-mutation prohibition. Gate strict manifest validation,
compiled small-system construction, forward operation, full-state fresh-session
checkpoint replay and learner compiler acceptance separately. Do not bypass a
failed learner gate or call a forward-only assembly a qualified training session.
One glial fragment is the initial supported schema; myelin, multiple-glia schema
extensions, measured extracellular anatomy and all-104 qualification remain open.

Phase result (2026-09-13): complete synthetic neuroglial session construction,
learner compilation, driven nonzero eligibility and fresh-session physical
replay passed. The 80-test affected gate covers existing session compatibility
and a corrected axial-resistivity cache: bare object-ID reuse had substituted
another quantity's resistivity. The bounded cache now retains and checks owners;
the chemical/electrical agreement gate was not relaxed. New validator/factory
coverage is 100%/97.06%. The diagnostic captures 100 finite arrays after 60
physical ticks and exactly replays them, including eligibility. Evidence and
remaining qualification are in `docs/biology/neuroglial-session-phase/README.md`.
