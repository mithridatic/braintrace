# BRAINCELL biology implementation evidence

Status on 2026-09-13: **partial implementation, not full H01/Example 21 completion**.
Worktree branch: `feat/h01-braincell-biology`, based on `e21e22c`.
The accepted contract is in [the specification](../specs/2026-09-11-h01-braincell-biology.md).
The [spatial assembly phase](spine-phase/README.md) now connects explicit spines
to the H01 builder and session factory, including fresh-session checkpoint replay.
The [neuron-glia phase](neuroglial-phase/README.md) connects actual neuronal
spikes to GABA/glutamate release, dynamic Kir cables and mapped astrocyte calcium.
The [measured glial assembly phase](glia-assembly-phase/README.md) reconstructs
one complete selected source fragment with all other fragments accounted for,
and verifies its 148-compartment Kir cable with exact fresh reconstruction.
The [tapered chemistry phase](tapered-chemistry-phase/README.md) maps that cable
to matching calcium/K volumes and conservative branch-junction diffusion,
including a bounded measured-fragment physical checkpoint replay.
The [neuroglial session phase](neuroglial-session-phase/README.md) adds explicit
extracellular/release manifests and builds the coupled model before learner
compilation, with active-eligibility fresh-session replay on a small fixture.
The [neuroglial learning phase](neuroglial-learning-phase/README.md) tests
finite-window encoder gradients through that session, BPTT/finite-difference
agreement and bounded diagnostic descent with independent input patterns.

## What is executable

| Mechanism | Implemented behavior | Qualification boundary |
| --- | --- | --- |
| Explicit spines | Topology-pinned H01 construction, cylindrical neck/head additions, source-location/region/contact remapping | Fixture and real 969-compartment source-component execution; additions explicitly synthetic, not a measured spine reconstruction |
| Stochastic release | Fixed Pr, uniform/linear/bell profiles, BrainState random streams, reset/replay, H01 contact-event masking | H01 session opt-in `h01-biology-release-v1`; no fitting of Pr |
| Extracellular transport | Matrix-free implicit finite-volume diffusion, uptake, fixed-bath and source accounting | Analytic, conservation, refinement and derivative fixtures; source-history erfc is a separate concentration-boundary mode |
| Potassium feedback | Dynamic intracellular/extracellular K and Nernst reversal, actual cable K-current exchange | Small initialized cable/network fixtures; membrane current sampled at step start; coupled time refinement at population scale pending |
| GABA | Actual inhibitory population spikes drive site selection, molecule delivery, uptake/diffusion and source Hill tonic receptor | Small coupled cable fixture; measured H01 site assembly pending |
| Astrocyte calcium | 25 states per segment, four radial shells, buffers, IP3, fixed ER reservoir, pump and coupled spatial solve | Local and three-segment independent NEURON fixtures; no demonstrated long-range calcium wave in measured H01 anatomy |
| Astrocyte K | Dynamic Kir4.1 cable, shared intracellular/extracellular K and conservative exchange with neuronal pools | Synthetic coupled fixture and independent fixed-pool NEURON cable reference; complete glial channel repertoire and measured H01 coupled assembly pending |
| Measured glial geometry | Source-pinned complete component selection, tapered paths, explicit exclusions and dynamic Kir cable construction | One 571-vertex H01 fragment, 148 CVs; complete session integration pending |
| Tapered calcium/K map | Integrated CV volumes and areas, four source shell fractions, conservative shared-junction diffusion | 148-CV calcium/K ledger and physical replay at 0.015 ms; diagnostic extracellular pools, no 3-D taper or long-wave qualification |
| Neuron-to-glia glutamate | Actual excitatory spikes drive explicitly dosed sites, extracellular transport and mapped IP3/calcium | Synthetic diagnostic dose, short calcium response against no-glutamate control; no physiological vesicle-dose or long-wave qualification |
| Myelin K | Explicit axon/sheath pools, radial transport, node bath boundaries, source pump laws | Conservative physical discretization; not exact source RxD equivalence, no coupled sheath electrical model |
| Learning | Finite-window chemical-feedback and synthetic full neuroglial session encoder diagnostics, BPTT directional checks and bounded descent | No recurrent contacts in the session fixture; ARC, stochastic-release gradients and joint myelin training remain unqualified |
| Physical waits | Exact .1 ms silent events, bounded compiled chunks, optional eligibility evolution, padding does not advance | Tiny durations exercised; 0/1/10/20-second counts tested, full-duration biological runs not performed |
| Physical checkpoints | v2 model/learner/chemistry/RNG/queue/cursor/optimizer capture, identity checks, atomic save and replay | Whole-event wait boundaries and already constructed matching sessions; no inside-cable-substep resume or automatic spatial reconstruction |

The existing baseline remains opt-in compatible. `H01Session.build` accepts
the release-only schema and `h01-biology-spines-v1`, including explicit contact
head targets and probabilities keyed by contact identity. The new
`h01-biology-neuroglia-v1` schema also constructs one selected glial fragment,
explicit extracellular fields and release maps. Multiple-glia and myelin
factory support remains pending. Unsupported spatial manifests, legacy
biological saves and biological topology mutation fail explicitly instead of
dropping/reusing unmapped state. Contact identity remapping is needed even when
counts agree.

## Reference and numerical evidence

Paper: [BRAINCELL preprint, version 3](https://www.biorxiv.org/content/10.64898/2026.08.22.746419v3.full).
Source: [RusakovLab/BRAINCELL](https://github.com/RusakovLab/BRAINCELL/tree/8acd1c23032a17af59ae1f813d142735fef8689c).
The [source manifest](reference/manifest.json) pins file hashes and paper hash;
the translated mechanisms retain the BSD-3-Clause notice in the package.

`neuron_reference.py` executes the original pinned `cadifus.mod` in independent
NEURON 8.2.7, not through the JAX implementation. The saved
[local trajectory](calcium-reference.json) covers all 25 states for 1 ms with
diffusion and pump disabled and matches at rtol 1e-7, atol 1e-12. The
[spatial trajectory](calcium-spatial-reference.json) covers three one-micron
segments for .2 ms, with nonuniform initial calcium and source parameter
readback. Comparison at the source .005 ms clock passes rtol .05, atol 1e-9.

Reaction/transport splitting failed the bounded spatial comparison. The coupled
backward-Euler implementation uses matrix-free Newton-Krylov and local 25-by-25
preconditioners, with an implicit coupled derivative. The selected .00125 ms
internal clock passes the strong-pulse comparison against .000625 ms at the same
5% tolerance. Four internal solves advance each .005 ms cable interval. Neither
that fixture nor line coverage establishes long-duration wave accuracy.

Source details preserved: bound mobile buffer diffuses longitudinally but not
radially; free mobile buffer and calcium diffuse in both directions; the source
IP3 surface/volume factor is two; ER is a fixed reservoir with an exchange
ledger. The local clamp is represented in the checkpointed IP3 state.

Unresolved source differences remain explicit:

- Paper results mention 500 synapses while methods mention 316 spines.
- Paper parameters differ from repository defaults.
- The pinned myelin source enables an axial diffusion coefficient despite a
  comment saying there is no longitudinal diffusion; radial rates also differ
  from the finite-volume shell resistance used here.
- `PERIAXONAL_PARAMS.md` and `Ko_Siphoning_Notes.md` are copied repository
  conversation notes, not independent scientific evidence.

## Anatomy acquired

The [official H01 release](https://h01-release.storage.googleapis.com/data.html)
provides the archive and public reconstruction layers. The full selected-neuron
SWC archive was downloaded and verified at SHA256
`3e0534df357ef2e92f6e0199962133cc9bc9733eb3a1fd6d1d315208ad63db47`.
Large inputs remain in the ignored `.cache/h01-biology` directory.

The c3 segment-property table identifies 5,128 labelled astrocytes, including
481 L2 entries. Candidate `63900941936` was selected by median L2 voxel count
for acquisition, **not** by demonstrated proximity to the selected neurons.
Its skeleton was acquired with physical nm coordinates/radii and archive hash
`6d424520d70c277e5b734b9ded86a8d0b839da391875733e63be1b20cf019e85`.
The [geometry audit](glia-geometry-audit.json) finds 104,133 vertices in 257
components; the largest has 79,152 vertices, or 76.01%. Keeping only that
component would discard almost 24%, so it is not silently adopted as a complete
astrocyte. Frame alignment, truncation, neighborhood selection and conversion
into an electrical/chemical discretization are unfinished.

The source astrocyte HOC additionally creates 10,000 large and 10,000 small glial
processes. It must not be labelled a purely measured human reconstruction.
The official parasitic-glia gallery example labels microglia/OPC, not astrocytes.
Acquiring these files does not establish matching glia or myelin for all 104.

## Runtime evidence and run limit

The authoritative current solver probe is
[calcium-runtime-coupled-128.json](calcium-runtime-coupled-128.json):
128 synthetic linear segments, four internal solves per tick, 10 cable ticks
(.05 ms biological time), CPU, valid result. Compilation plus first call took
2.93 s; the warmed call took **1.86875 s**, RSS approximately 587 MB.
It ran separately from the test gate.

Linear-in-duration extrapolation **for that same small fixture only** gives
approximately 10.38 hours per one biological second and 8.65 days for 20 seconds.
This is not an all-104 runtime forecast or a long-trajectory stability result.
Earlier `calcium-runtime-128.json`, `calcium-runtime-1024.json` and
`calcium-runtime-1024-adaptive.json` used superseded split solvers and must not
be used to estimate the current solver. The first 1024 probe also overlapped
tests and is not an isolated performance measurement.

The pre-existing primary-worktree
`docs/evidence/h01-104-corrected-initialization-decision.json` records successful
baseline construction and initialization for **807,588 compartments**, at
291.165 s and 144.223 s, respectively, with peak RSS 9.35 GB. Its compiled
10 ms runtime is `UNTESTED_TIMEOUT`; this implementation did not rerun or qualify
that baseline. Initialization should no longer be described as memory-blocked.

No run longer than the authorized 15-minute local limit and no paid run was
launched. The long physical waits require a measured performance redesign or a
separately approved longer run after a complete model is assembled. A longer
run by itself does not resolve the unfinished integration below.

## Verification and corrections

`verification.xml` and `coverage.json` record the first milestone's affected CPU
gate; `verification-summary.json` pins that milestone's totals and implementation
hashes. The subsequent spine-phase directory holds its own current gate records.
Coverage is reported for new production modules separately from tests and the
unrelated pre-existing Example 21 code. This is not a full-repository or GPU gate.

Important regressions covered: tiny K currents surviving the implicit solve;
positive pool mapping at cable boundary nodes; GABA channel point shapes;
random-key exclusion from differentiable factors while retaining forward state;
coupled-calcium reverse-mode derivatives and clamp state; reset/replay through
the actual H01 learner; refusing stale biological maps during mutation; and
rejecting reservoir graphs in a closed intracellular solve.

The lessons are explicit conservation checks at small physiological fluxes,
actual cable integration tests rather than formula tests alone, typed-key-aware
dtype inspection, and validating equation-domain assumptions at the API boundary.
Gradient properties use the finite-window oracle where appropriate. The readout
state and older opaque-scan warnings emitted by the broader gate must not be
represented as warning-free joint-biological learning evidence.

## Remaining implementation sequence

1. Extend the implemented spine/neuroglial manifest and H01 session
   factory to multiple glial fragments and myelin. Select and qualify local structures using the
   coordinate/proximity audit, preserving all exclusions and synthetic fallbacks.
   Spine contact, probe and electrical-region remapping is now executable;
   general biological topology mutation and optimizer transport remain pending.
2. Qualify the now-executable neuronal spike/release, astrocyte electrical K
   buffering and glutamate/calcium session factory on measured source/domain placements.
   Assemble myelin/sheath membrane dynamics with species and reservoir accounting.
3. Resolve source myelin discretization with independent NEURON experiments;
   test long calcium-wave propagation, coupled timing and spatial refinement.
4. Optimize and measure the assembled 4/12/40-cell progression before estimating
   104-cell execution. Never bypass the physical clock to fit a run budget.
5. Integrate waits before ARC answer requests, matched 0/1/10/20-second controls,
   full spatial reconstruction on restore, topology remapping and cable-substep
   continuation. Keep biology frozen; train encoder/contact/readout only.
6. Qualify joint sparse pp-prop factors and actual learning, then execute the
   all-104 forward, replay and bounded training gates within approved resources.
   Preserve exact spike-count, 1 ms timing, 1 mV voltage and .05 ms phase criteria.

The overall plan remains **incomplete** until these integration and population
gates pass. The current mechanism tests do not establish human physiological
qualification or Example 21 readiness for the full BRAINCELL mechanism set.
