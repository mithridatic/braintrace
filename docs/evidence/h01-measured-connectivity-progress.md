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

Historical: the initial spatial check resolved 79 records to released morphology IDs at the
time; the resolved edge list of 2026-09-06 shows all 104 cells identity-consistent in the
proofread volume (`h01-resolved-edge-list.json#/cells`, 104/104 `identity_consistent`, no C3
label shared by two cells).
Each accepted identity has at least three distinct sampled voxels with the
same nonzero cell label. Other samples must be background or that same label.
Background supplies no identity evidence. The remaining 25 records needed more
checks at the time (since resolved, see above). Different release IDs must not be joined by numerical equality alone.

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
It is not endpoint-verified. The batch 5-voxel re-verification of all 123 candidates
stalled (six hours without completing its first batch of ten edges); the one-edge-at-a-time
recheck below completed it (8 of 8 edges, 17.4-18.9 s each, none over the 300 s budget;
`h01-endpoint-recheck-5voxel.json`). 54906016 remains unverified at 5 voxels
(`both_within_box` false). The reciprocal wiring in the circuit therefore stays illustrative and
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

## 3,000-sample rescan (2026-09-07): first launch stopped (historical; completed the same day, see below)

`h01_c3_edge_list.py --limit 3000` ran detached with per-cell checkpoints and completed
26 of 104 cells at 61.2 to 113.9 s each, then produced no progress line for 17 minutes
(a network stall on the volume, the same signature as the 2026-09-06 batch) and was
killed under the fail-fast rule. The checkpoint `h01-resolved-edge-list-3000.cells.json`
held the 26 finished cells; the rescan then resumed from that checkpoint under the watchdog
and completed 104 of 104 cells in two launches (15:28:12 to 18:28:53; 7,268.3 s + 3,571.9 s
wall, one 600.7 s stall kill), so the checkpoint now holds 104 cells
(`h01-c3-scan-3000.watchdog.json#/status` complete, `#/cells_scanned` 104). A 6,000-sample
attempt earlier the same day ran at 176 s per cell and was stopped after one cell as
over budget. `full_export_scanned` stayed false (export coverage, not this scan); the released
1,500-sample list stood until the completed rescan superseded it.

## SP7 connectivity completion tooling (2026-09-07)

Spec: [connectivity completion](../specs/2026-09-07-h01-connectivity-completion.md).

- **Watchdog (SP7a).** [`h01_c3_scan_watchdog.py`](h01_c3_scan_watchdog.py) resumes
  `h01_c3_edge_list.py --limit 3000` from `h01-resolved-edge-list-3000.cells.json` (26 of 104
  cells when this entry was written; now 104 of 104, key `{"limit": 3000, "archive": "proofread104.zip"}`), kills the child after 600 s
  without a `cell` line, relaunches at most 3 times, and records every launch with wall clocks.
  Launcher [`h01_c3_scan_watchdog.ps1`](h01_c3_scan_watchdog.ps1) (Start-Process, detached).
  One-cell dry run through the launcher on `2530864375` (fresh output name, so the 3,000
  checkpoint is untouched): 70.0 s wall, exit 0, no kill, 33 C3 ids, 0 edges
  ([record](h01-c3-scan-dry-run-3000.watchdog.json)). The full resume (78 cells at 61-114 s
  each in the checkpoint, derived 1.3-2.5 h) had NOT been run when this entry was written; it
  was approved and ran the same day (two launches, 7,268.3 s + 3,571.9 s wall, 104 of 104;
  see the completed-rescan section below).
- **Export coverage (SP7b).** The export prefix holds **166 shards, 32.86 GB**
  ([listing](h01-c3-export-listing.json)); 9 are local and all nine were already audited, so
  `full_export_scanned` is false as **"9 of 166 shards scanned"**
  ([coverage](h01-export-scan-coverage.json), now the source of the summary's flag). Measured:
  the audit's record pass over one local shard takes 33.0-47.9 s (nine repeats, mean 38.4 s,
  repeat range 14.9 s); one shard download 197.6 MB in 5.45 s (36.3 MB/s, single
  measurement). Derived for the 157 remaining shards: 6,029 s read + 857 s download = about
  1.9 h, over the 15-minute rule, so not run.
- **Merge check (SP7c).** [`h01_pair_merge_check.py`](h01_pair_merge_check.py) on the pair
  `4157825456` / `5654281423` (71 candidates): no C3 label shared in the edge list or the
  3,000 checkpoint; 70 of 71 rows read background at both endpoints, 1 reads the pre cell
  only; 0 rows with one label at both ends; 0 within the box. **Verdict: suspected merge
  undetermined** ([json](h01-pair-merge-check.json), [table](h01-pair-merge-check.md)). The
  decisive online step is reading the C3 label at the 142 endpoint voxels against each cell's
  sampled `c3_ids`.
- **Recheck (SP7d).** The rescan ran; a 10-edge 5-voxel recheck of the diff was executed
  (`h01-endpoint-recheck-5voxel-3000.json`, 10 of 10 completed): none of the 3 new candidates
  (101412196, 127313472, 143320914) is verified or within the box, and the 7 previously in-box
  edges are unchanged, so the `prepare_connectivity` trigger did not fire because no new verified
  edge exists (`h01-resolved-edge-list-3000.json#/counts/verified_edges` 3).

Coverage state at the time of this entry: 26 of 104 cells at 3,000 samples; 9 of 166 export
shards. Superseded by the completed rescan below.

## 3,000-sample rescan completed (2026-09-07, SP7a/SP7d)

The rescan was resumed from the 26-cell checkpoint under the watchdog
([record](h01-c3-scan-3000.watchdog.json), [launch logs](h01-c3-scan-3000.launch-logs.json),
[result](h01-resolved-edge-list-3000.json), [checkpoint](h01-resolved-edge-list-3000.cells.json)).
Coverage: **104 of 104 cells scanned at 3,000 samples** (status `complete`).

| Quantity | Value |
| --- | --- |
| Launches | 2 (launch 1: 15:28:12 to 17:29:21, 7,268 s, pid 38508, 26 to 78 cells, **killed after 600.7 s without a progress line**; launch 2: 17:29:21 to 18:28:53, 3,572 s, pid 36472, 78 to 104 cells, exit 0) |
| Kills / relaunches | 1 stall kill, 1 relaunch; the cell in flight at the kill was rescanned by launch 2, so nothing was lost |
| Wall clock per cell (all 104, checkpoint `seconds`) | median 118.5 s, range 61.2 to 168.8 s |
| Wall clock per cell (78 cells under the watchdog) | median 125.1 s, range 73.3 to 168.8 s |
| Index query and endpoint pass (launch 2 after cell 104) | included in the child's 3,564.2 s wall |
| Cells identity-consistent | 104 of 104; no C3 label shared by two cells |
| C3 segment ids attributed | 14,145 (1,500 samples: 10,458) |
| Candidate directed contacts | 126 (1,500 samples: 123) |
| Directed cell pairs | 33 (1,500 samples: 31) |
| Endpoint-verified contacts | 3, the same three: 8105899, 124698307, 65017731, records byte-identical |
| Contacts within the 2-voxel box | 7, the same seven |
| Common candidates whose record changed | 0 of 123 |

**New candidates (SP7d diff against `h01-resolved-edge-list.json`)** and their 5-voxel
recheck ([json](h01-endpoint-recheck-5voxel-3000.json), one edge at a time, 300 s budget,
17.7 to 30.1 s per edge, no budget exceeded):

| Annotation | Pre cell | Post cell | Type | 2-voxel box | 5-voxel recheck | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 101412196 | `5584343344` (I) | `4188575291` (E) | 1 | pre exact, post background | pre `[0,0,0]`, post none in the box | not verified; new I-to-E pair |
| 127313472 | `4157825456` (E) | `5654281423` | 2 | pre exact, post background | pre `[0,0,0]`, post none in the box | not verified; 72nd candidate on the suspected-merge pair |
| 143320914 | `1684504313` (I) | `883843993` (E) | 1 | pre background, post exact | pre none in the box, post `[0,0,0]` | not verified; new I-to-E pair |

No new candidate becomes endpoint-verified or within-box at either tolerance, so the
`prepare_connectivity` trigger did not fire: the topology SP8 builds from
([h01-verified-network.json](h01-verified-network.json)) is unchanged, with 3 verified contacts,
2 construction-ready, 6 incident cells. `h01-verified-network-3000.json` was not produced
because the verified edge records and every cell's `identity_consistent` flag are identical in
the two lists, so the selection input is the same. The 7 previously in-box edges rechecked
with the same offsets as on 2026-09-07 (the 3 exact contacts exact; 95907584 post offset 1 in
x; 108171241 pre offset 1 in x; 125872198 post offset -2 in x; 95566965 pre +1, post -1 in z).

Doubling the sample count raised the attributed C3 ids by 35 % and added three candidates
but no verified contact. Coverage state: 104 of 104 cells at 3,000 samples; 9 of 166 export
shards. Absence of a contact is still not established: 3,000 samples per cell do not cover
every C3 segment, and `full_export_scanned` stays false.
