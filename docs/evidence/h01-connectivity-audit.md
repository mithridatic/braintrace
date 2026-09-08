# H01 circuit connectivity audit

## Proofread-104 table

The checksum-pinned local table has 264090 rows and 264084 unique pairs of
pre- and postsynaptic coordinates. No exact pair has both a presynaptic and
postsynaptic cell assignment. Therefore, matching repeated coordinates in this
table supplies no candidate partner link between the 104 cells.
This is not evidence that these neurons have no connections in the tissue.
It is a limit of this table and matching method.
The [audit record](h01-partner-coordinate-audit.json) records the rule and counts.

## Broader C3 release

The [official release page](https://h01-release.storage.googleapis.com/data.html)
lists a C3 synaptic connections database and recommends C3 for connectomic analysis.
Its IDs belong to the C3 segmentation. Do not assume that they are proofread-104 IDs.

The [bounded metadata audit](h01-c3-metadata-audit.json) records the public URLs,
object metadata, range checksum, and one complete sample record.
The 16384-byte range contains 21 complete records from a 757111597-byte JSON shard.
The complete shard was not downloaded.
The sample has explicit `pre_synaptic_site.neuron_id` and
`post_synaptic_partner.neuron_id` fields. It also has site IDs, base neuron IDs,
centroids, compartment labels, a top-level type, and confidence.
The postsynaptic neuron ID and base neuron ID differ in the sample.
Thus, raw equality between different ID fields is not a valid general join rule.

The top-level synapse type must be decoded from its schema before assigning E/I.
Site-level type fields are separate and must not be used as synapse E/I labels.
Confidence is a source value. It is not a manual verification flag.

## Required join evidence

Before using a measured C3 link with a proofread-104 model:

1. Establish the release-specific ID mapping, including split and merge cases.
2. Verify pre- and postsynaptic sites against the mapped cell geometry.
3. Retain site IDs, source records, coordinate units, and rejection reasons.
4. Decode synapse type from the documented source convention.
5. Keep conductance, delay, and receptor kinetics separate from anatomy.

No link currently passes this complete join check.
If the initial small circuit uses illustrative wiring, label every edge as inferred.
Do not call that circuit an H01 reconstructed microcircuit.
