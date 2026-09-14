# Explicit neuroglial H01 session factory

This phase adds `h01-biology-neuroglia-v1` to `H01Session.build`, following
`fd780a9` on `feat/h01-braincell-biology`. The first schema supports the active
neuronal topology, optional explicit spines, and one selected glial fragment.
Multiple-glia schemas and myelin electrical construction remain future work.

## What the manifest controls

The manifest pins the complete nested spine/topology declaration; glial source
selection and electrical settings; CV geometry hashes and every membrane's
extracellular index; extracellular centers, accessible volumes and explicit
transport/clearance/boundary rates for K, GABA and glutamate; initial K pools and
temperature; tonic receptor settings; calcium rates and internal substeps; and
release source identities, destination volumes, site counts, doses and seeds.
Extracellular origin and modeling basis are explicit. None of these placements
is inferred from nearest-neighbor proximity. Destination indices represent the
declared finite-volume release locations, not a reconstructed synaptic cleft.

`manifest.json` is an executable **synthetic diagnostic example**. It is not a
measured extracellular reconstruction or a physiological glutamate-dose claim.
Centers are recorded coordinates; transport conductances and CV assignments are
supplied explicitly rather than inferred from those centers. Altering any rate
or placement changes the canonical biology identity.

Build neurons with environment K, paint tonic GABA before initialization, and
load the independent glial cable from its source digest. The factory reconstructs
exact chemical geometry and rejects stale hashes or map cardinalities before
binding chemistry. It then compiles the sparse learner with the coupled model.
The glia remain outside the neuronal ARC encoder/readout population.

## Construction and checkpoints

Pass the manifest in `settings['biology']`, and glial assets as a mapping from
source SHA256 to local path:

```python
session = H01Session.build(
    topology, archive, trainer_type,
    settings=settings,
    biology_assets={glial_source_sha256: glial_source_path},
)
```

Prepare CV maps by inspecting the declared source cables with
`CableChemicalGeometry`; author extracellular mappings explicitly, then build
from the final manifest. The factory does not accept placeholder geometry hashes.
Local paths are asset resolution, not biological identity. No implicit download
or omitted-glia fallback occurs when an asset is missing.

Physical v2 snapshots automatically use the session's canonical neuroglial
manifest. A conflicting explicit manifest or settings changed after construction
is rejected against the chemistry's construction identity. To resume, reconstruct
the same session from its settings and verified local assets, create its matching
`PhysicalWait`, then call `restore_physical`. Episode-only biological checkpoints
and biological topology mutation remain explicitly unsupported.

## Diagnostic evidence and boundaries

Run `python -m docs.biology.neuroglial_session_probe` from the checkout root in
the scientific development environment. It uses the existing co-located importer
fixtures with synthetic source bytes and explicit synthetic glia/domain geometry.
The development-only archive checksum substitution is scoped to this fixture;
the production factory retains source hash checks.

The diagnostic has one seven-compartment E neuron and one glial compartment.
One driven event followed by two silent events advances 60 cable steps at
0.005 ms, or 0.3 ms total. The driven event produced 119 nonzero eligibility
entries. All 100 packed physical/learner arrays were finite and exactly matched
after a fresh-session reconstruction, checkpoint restore and continuation.
The glutamate release stream is enabled; this probe does not establish evoked
release physiology. Earlier coupled-cable tests separately exercise actual
GABA/glutamate release and receptor/calcium responses.

The gate distinguishes assembly, learner compilation, active eligibility
execution and exact replay from gradient accuracy, optimizer descent and human
physiology. It is not a joint-learning accuracy or all-104 qualification.
The prior readout-state warnings remain visible.

The final affected CPU gate passed **80 tests**. New-module statement coverage
is **100% for the manifest validator** and **97.06% for the assembly module**.
The gate emitted 11 warnings: ten readout-state warnings and one float64-to-
float32 warning from an existing construction test. This is not a warning-free,
GPU or full-repository gate. `verification-summary.json` pins the corrected
implementation and final diagnostic artifacts.

## Edge cases and next work

Tests cover unknown manifest fields, source polarity and row mismatches,
incorrect centers/edges, omitted membranes, stale geometry/destination indices,
unaccounted K uptake, invalid clocks/receptors/calcium, missing source assets,
conflicting snapshots, and postconstruction manifest mutation. The latter
regression exposed a snapshot-labeling gap: comparison now uses the immutable
construction hash rather than trusting the mutable settings dictionary alone.

The broader run also exposed an older axial-resistivity conversion cache keyed
only by Python object ID. After object-address reuse it could return a different
cable's resistivity; the chemical/electrical geometry check correctly rejected
the resulting mismatch. A deterministic collision test reproduced 150 ohm cm
returning 100 ohm cm. The cache now verifies and retains each owner, with a
1024-entry bound because Quantity objects do not support weak references.
Construction parity and lifetime tests cover this fix. Geometry tolerances were
not relaxed. Earlier artifacts remain historical; this phase's verification
pins the corrected construction implementation.

Next work is qualification with explicitly justified measured source/domain
placements, finite-window coupled learning oracles and bounded population/runtime
progression. Long calcium waves, spatial/time refinement, multiple glia, myelin
electrophysiology and full-population physiology remain unqualified. Each local
probe is bounded by the existing 15-minute limit.
