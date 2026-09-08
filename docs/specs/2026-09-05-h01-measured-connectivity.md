# H01 measured connectivity

## Required behavior

Replace illustrative circuit edges with contacts from the released synapse
records where both endpoint identities can be resolved. Do not assume that
C3 segment IDs equal the IDs in the 104-cell morphology archive.

## Source and identity checks

- Read the public `proofread104_neurons_20210511.zip` archive from
  `https://storage.googleapis.com/h01_paper_public_files/`.
- Preserve each source filename, metadata, base-segment membership, and hash.
- Check archive identities against the released proofread volume. Use several
  source base locations. Keep missing and conflicting labels unresolved.
- Join synapse `base_neuron_id` values to unique proofreading records.
- Exclude ambiguous ownership and base segments with recorded merge cuts until
  endpoint-level spatial checks resolve them.
- Preserve the full source synapse record, export shard, and record index.
- A partial export scan proves only the contacts found. It does not prove that
  other connections are absent or that contact counts are complete.

## Circuit boundary

Use source endpoint coordinates to place contacts on the selected morphology.
Check component membership and placement distance before construction. Do not
move a missing dendritic contact to the soma without explicit provenance.
Keep conductance, delay, and channel parameters marked as model assumptions.
Source synapse detection is not proof of manual synapse verification.

## Checks

Reject ambiguous cell mappings, missing partners, duplicate synapse records,
and invalid coordinates. Verify direction from pre/post source records. Compare
connected and edge-removal controls with identical cells and inputs. Test the
importer independently of network access with small source-shaped fixtures.

The initial audit is read-only with respect to the model. It must establish
supported contacts before the circuit defaults change.

## Selected anatomical candidate

Annotation `8105899` supplies an I-to-E contact from proofread cell `5584343344`
to `4157825456`. Its C3 endpoints are `85384868035` and `40781762246`.
Direct voxel queries in both segmentations agree with these identities.
The source type is 1 (inhibitory), consistent with the presynaptic interneuron
tag. Neither proofreading record has a recorded base-segment merge cut.

After cable placement passes, use component 0 of these cells for the measured
two-cell example. Preserve the selected E and I channel profiles. Include only
this supported directed contact. Do not add reciprocal wiring for symmetry.
Use the measured axon endpoint for the presynaptic output location and the
measured postsynaptic endpoint for receptor placement. Preserve borrowed delay,
conductance, and receptor kinetics as explicit assumptions.

Keep the prior reciprocal configuration available only as an explicitly
illustrative diagnostic. It must not be the measured example's default.
An explicit disconnected control must retain the same cells and placements.
