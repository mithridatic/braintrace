# H01 contact search

The circuit default now uses the anatomically supported I-to-E contact below.
It does not represent the complete H01 network. Human physiology and actual
H01 event-delivery validation remain open.

## Identity evidence

The [authors' edge-list code](https://github.com/ashapsoncoe/h01/blob/main/get_edge_list_for_pr_cells_and_unproofread_targets.py)
uses base-segment membership from the proofreading records. The public archive
is `gs://h01_paper_public_files/proofread104_neurons_20210511.zip`.
Its SHA-256 is `364195428726bb09e1674323ffb66d280acf9de28f866b9e21bd2f86f0914881`.

The archive contains 104 records and 365,404 unique base segments. No base
segment has two cell owners. It contains no entries in `all_syn` or the
incoming/outgoing `verified_synapses` fields. Morphology proofreading does
not establish manual synapse verification.

The initial spatial check resolves 79 records to released morphology IDs.
Each accepted identity has at least three distinct sampled voxels with the
same nonzero cell label. Other samples must be background or that same label.
Background supplies no identity evidence. The remaining 25 records need more
checks. Different release IDs must not be joined by numerical equality alone.

## Synapse evidence

Nine export shards contain 8,991,719 records. Of these, 39 have both endpoint
base segments in the proofreading archive. All 39 refer to the same
proofreading record at both ends. None is an accepted connection between two
different cells. This partial search does not prove that other edges are absent.

The released per-cell annotation index provides a second search path. Its
endpoint order, coordinates, and type were checked against an export record.
Querying the 17,594 axon base IDs as candidate C3 search keys returned 2,699
annotations. Query-key equality does not establish cell identity. The endpoint
check uses the proofread volume for that purpose.

The endpoint check found two candidates with different cell labels:

| Annotation | Presynaptic cell | Postsynaptic cell | Cell tags |
| --- | --- | --- | --- |
| 8105899 | 5584343344 | 4157825456 | Interneuron to pyramidal |
| 124698307 | 4188575291 | 3955003482 | Pyramidal to pyramidal |

Both endpoint pairs were read again through the annotation `by_id` index.
The coordinates and type codes match the related-object records exactly.
The I-to-E candidate has C3 IDs `85384868035` and `40781762246`. These differ
from the proofread-cell IDs above, which come from spatial segmentation checks.
Its nearest source samples lie in component 0 of each cell, at distances
0.1773 and 0.1339 micrometers. The cable projection also passes: the distances
are 0.1557 and 0.1331 micrometers. Both placements use the soma-containing
connected component 0. The complete source and placement record is
[saved here](h01-ie-measured-contact.json). The circuit implementation uses these sites. Synthetic-fixture delivery
checks pass; actual H01 event-delivery checks remain open.

The [authors' E/I analysis](https://github.com/ashapsoncoe/h01/blob/main/get_neuron_e_to_i_ratios.py)
uses top-level type 1 for inhibitory and type 2 for excitatory synapses.
The candidate codes agree with the presynaptic cell tags. This agreement does
not establish receptor kinetics, conductance, or manual review.

See [the shard summary](h01-measured-connectivity-summary.json) and the
[index audit script](h01_index_connectivity_audit.py). The annotation decoder
follows the [Neuroglancer format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md).

## Remaining circuit checks

For each candidate, verify the directed partner pair and place both endpoints
on the correct connected morphology components. Retain the source synapse ID
and placement errors. Resolve segment cuts before accepting affected contacts.
Then run matched edge-removal controls. Conductance, delay, and channel values
remain borrowed parameters; anatomical contact evidence does not validate them.
