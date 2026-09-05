# Reusable H01 import into BrainCell

## Objective

Import measured H01 human neuron morphologies into BrainCell through a public,
installable API usable by independent projects and PP-Prop examples. Keep the
importer separate from ARC task encoding, decoders, and training orchestration.

## Contract

- Provide `braintrace.datasets.h01` with explicit download/cache, neuron listing,
  and morphology loading operations. Imports must not initiate network access.
- Use the official 20210601 `proofread_104` SWC archive, with its CC BY 4.0
  attribution, source URL, release identifier, and archive SHA-256 recorded.
- Inspect the actual archive before choosing coordinate conversion or repair
  policies. Validate topology, finite coordinates, positive radii and units.
  Any necessary conversion must be explicit and preserve original source bytes.
- Return BrainCell morphology objects with source identity and import diagnostics.
  Leave biophysical parameters and discretization configurable by the caller.
- Demonstrate constructing and running a multicompartment BrainCell cell from a
  real downloaded H01 neuron. Label demonstration electrical properties as
  assumptions, not measurements from that neuron.
- Provide reusable usage instructions, including use outside the repository and
  the boundary between anatomical import and PP-Prop learning qualification.
- No changes to Example 21 behavior. No claim to import all synaptic connectivity,
  reconstruct branches outside the tissue, fit human physiology, or establish
  PP-Prop correctness from a forward simulation. These require separate evidence.

## Verification

Co-located offline tests cover cache integrity, archive selection, missing cells,
malformed data and source provenance. Exercise the real H01 archive and compiled
BrainCell simulation; record source checksums, imported cell identity, geometry
and finite voltage results. Require greater than 90% coverage of the new importer
and verify its public modules appear in a built wheel.

## Sources

- https://h01-release.storage.googleapis.com/data.html
- https://brainx.chaobrain.com/braincell/tutorials/cell.html

## Source inspection decisions

The actual archive has 104 cell IDs and 3,327 disconnected component files.
Positions are 32 x 32 x 33 nm voxels, radii are physical nanometers, and H01
annotation codes differ from standard SWC types. Preserve all original rows;
normalize geometry and IDs for BrainCell using neutral custom branch types.
Do not guess missing labels or join separate components. Load requires an
explicit component ID and rejects singleton components without cable geometry.
Cell 810151953 has one archived component and is the reproducible full-cell
import demonstration. Its uniform passive dynamics are explicitly hypothetical.

The initially proposed pyramidal-cell label is not asserted by the SWC archive;
the demonstration therefore uses an identified H01 cell without inferring a
cell type from morphology alone.

## Regression reflection

A reproducing test showed that an archive replaced after construction could
bypass the initial checksum. Verify the checksum before each component load,
not only when opening or downloading the archive. Keep this regression test.

## Authorization

The user requested a reusable H01 import after the proposed anatomy import,
electrical-model construction, and validation approach. Implementation occurs on
`feat/h01-braincell` in an isolated worktree.
