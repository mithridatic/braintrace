# Useful H01 annotations for BrainCell

## Authorization and intent

The user approved extending the imported cells after the recommendation to use
compartment labels first and synapse locations next. Keep the implementation
reusable from installed BrainTrace, independent of Example 21 and PP-Prop.

## Requirements

1. Decode every released SWC annotation code, preserving unknown values and
   distinguishing source annotations from adapter inference. Expose annotated
   node locations and selectable cable regions without changing source geometry.
   Strict regions require agreeing endpoint labels. An explicitly selected
   sample-neighborhood policy may extend each sample to half its incident cable;
   this is an inference, never a verified compartment boundary.
2. Provide a soma recording/stimulation location from an actually soma-labelled
   sample, with an explicit selection rule and no fallback to the arbitrary root.
3. Load the official proofread_104 segment properties: layer/cell-type tags,
   measurements, descriptions, and E/I aggregate counts. Allow selection by tags.
   Pin checksums and preserve source provenance; no network at package import.
4. Load proofread_104/synapse_locations.csv and project the appropriate pre/post
   endpoint onto the selected component's cable. Return source row identity,
   both endpoint coordinates, projection distance, and unmatched/ambiguous status.
   Require an explicit maximum projection distance. Never join components.
5. Preserve the distinction between these releases: the demo cell has 388
   incoming synapses in segment properties but 810 post rows in the CSV.
   Do not reconcile by deleting rows or assign the aggregate E/I labels to
   individual CSV rows. CSV has no synapse ID, E/I type, or partner neuron ID.
6. Supply installed-library examples and an enriched demo using soma and actual
   synapse positions with caller-selected electrical assumptions. Demonstrate
   compiled simulation, useful region painting, and source-backed metadata.
7. Co-located meaningful tests, >90% coverage for changed dataset modules;
   real-release validation, installation check, and written evidence.

## Source discoveries

Official base: https://storage.googleapis.com/h01-release/data/20210601/proofread_104/

- `segment_properties/info`: gzip transported JSON; decompressed SHA-256
  `d8b9f54822460d4eb43fd7ca61897852a78c819bf229c8f27396c186f762716c`.
- `synapse_locations.csv`: 264090 rows, 18917333 bytes; SHA-256
  `640ccb12c75b930f96373c7273bbf9688aa111b97b425ee22d8ef6db19f496be`.
- CSV columns: `104,prepost,x,y,z,prex,prey,prez,postx,posty,postz`.
  Coordinates are in the synapse grid (8,8,33 nm), distinct from SWC positions.
- Cell 810151953 has tags L2, pyramidal, neuron; 31 soma-labelled samples.
- Published site advertises subcompartments/info but that URL returns 404;
  use the actual SWC codes and do not claim a volumetric annotation import.

## Boundaries and validation

No fitted human electrophysiology, automatic receptor assignments, reconstructed
missing axon, PP-Prop training qualification, or complete circuit emulation.
Proofread morphology does not establish per-label manual verification. Every
annotation reports that per-item review is not supplied. Tests cover wrong
units, unknown codes, sparse/conflicting labels, absent soma, branch junctions,
ambiguous projections, distance rejection, corrupt assets, malformed metadata,
and mismatch between whole-cell metadata and selected component geometry.

## Validation findings

BrainCell clones morphology during initialization. Selection guards therefore
compare geometry/topology fingerprints rather than Python object identity;
compiled integration tests reproduced the original guard failure before its fix.

The annotated CLI defaults to 64-bit precision, without changing the library
caller's global environment. A full-cell zero-input control at this precision
showed up to 0.00241001135 mV drift. An additional 1e-6 mV equilibrium probe
failed; that stricter numerical accuracy is not established by this feature.
Preserve this result in the evidence and usage guide. The deliverable remains
source-backed annotations and placement, with demonstrated forward execution,
not a change to BrainCell's solver or a qualified human electrophysiology model.
