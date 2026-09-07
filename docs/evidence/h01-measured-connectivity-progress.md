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

## Population edge list (2026-09-06)

The C3 relationship index was queried for all 104 proofread cells
([script](h01_c3_edge_list.py), [result](h01-resolved-edge-list.json),
[per-cell checkpoint](h01-resolved-edge-list.cells.json)). Up to 1,500 skeleton
nodes per cell were sampled into C3 labels; every cell was identity-consistent
in the proofread volume (one nonzero label at its samples, no C3 label shared
by two cells). The scan took 950.5 s of wall time after a resume from the 93rd
cell; the first attempt stalled on name resolution.

| Quantity | Value |
| --- | --- |
| Cells scanned, identity-consistent | 104 of 104 |
| C3 segment ids attributed to the 104 cells | 10,458 |
| Candidate directed contacts between two different cells | 123 |
| Directed cell pairs | 31 |
| Cells with at least one candidate contact | 34 |
| Contacts with both endpoints read as the expected cell at the annotation voxel | 3 |
| Contacts with both endpoints within a 2-voxel box | 7 |
| Cells joined by an exactly verified contact | 6 |

Verified exactly: 8105899 (I `5584343344` to E `4157825456`, type 1),
124698307 (`4188575291` to `3955003482`, type 2) and 65017731 (`3519995546` to
`3680152874`, type 2). The reverse contact 95907584 (`3680152874` to
`3519995546`, type 1) is within the box, so that pair is the only reciprocal
pair with endpoint support.

**E-to-I candidate 54906016** (`4157825456` to `5584343344`, type 2) is the
only candidate in that direction. Its postsynaptic endpoint reads the I cell one
slice away; its presynaptic endpoint reads background inside the 2-voxel box.
It is not endpoint-verified. A 5-voxel re-verification of all 123 candidates
was launched and stopped after six hours without completing its first batch of
ten edges (network stall on the proofread volume), so the wider tolerance is
untested. The reciprocal wiring in the circuit therefore stays illustrative and
opt-in.

**Caution on the largest pair.** 71 of the 123 candidates join
`5654281423` and `4157825456` in both directions with mixed type codes, and
none of them has an endpoint inside the box. Mixed type codes on one directed
pair and zero endpoint support are consistent with a segmentation merge or a
label collision rather than 71 synapses; these are counted as candidates only.

This list is a lower bound: 1,500 samples per cell do not cover every C3
segment of a cell, and absence of a candidate does not establish absence of a
contact. It does not qualify any physiology.

## Remaining circuit checks

For each candidate, verify the directed partner pair and place both endpoints
on the correct connected morphology components. Retain the source synapse ID
and placement errors. Resolve segment cuts before accepting affected contacts.
Then run matched edge-removal controls. Conductance, delay, and channel values
remain borrowed parameters; anatomical contact evidence does not validate them.

## 5-voxel endpoint re-check (2026-09-07)

Rerun one edge at a time with a 300 s budget per edge (`h01_endpoint_recheck.py`,
box 5 voxels in x and y, 2 in z); all 8 edges completed in 17 to 19 s each, so the
six-hour stall of 2026-09-06 was the batch, not the volume. Result
(`h01-endpoint-recheck-5voxel.json`): the E-to-I candidate 54906016 is **not
verified at 5 voxels either**: its presynaptic endpoint reads background throughout
the box and its postsynaptic endpoint reads the I cell one slice away. The
reciprocal wiring therefore stays illustrative as a tested negative, not an
untested one. The reciprocal pair 95907584 has both endpoints within one voxel
(pre exact, post offset 1 in x); 108171241, 125872198 and 95566965 are within 1 to 2
voxels; the three exact contacts are unchanged. Offsets of one voxel or one slice
are candidates for a relaxed acceptance rule, which is a decision for the network
builder, not this record.

## 3,000-sample rescan (2026-09-07): stopped, untested

`h01_c3_edge_list.py --limit 3000` ran detached with per-cell checkpoints and completed
26 of 104 cells at 98 to 110 s each, then produced no progress line for 17 minutes
(a network stall on the volume, the same signature as the 2026-09-06 batch) and was
killed under the fail-fast rule. The checkpoint `h01-resolved-edge-list-3000.cells.json`
holds the 26 finished cells and the run resumes from it with the same limit; a 6,000-sample
attempt earlier the same day ran at 176 s per cell and was stopped after one cell as
over budget. `full_export_scanned` stays false; the released 1,500-sample list stands.
