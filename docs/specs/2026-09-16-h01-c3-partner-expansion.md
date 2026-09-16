# C3 partner expansion: a connected human E/I core for Example 21 (2026-09-16)

Status: approved by J 2026-09-16, registered before any run. Parent records:
[keep/drop](2026-09-16-h01-keep-drop.md) (17 of 104 kept, 0 interneurons, 0 kept
contacts), [driven window and anatomy transfer](2026-09-15-h01-driven-window-and-anatomy-transfer.md)
(Y7), [per-cell human fit](2026-09-15-h01-per-cell-human-fit.md) (proposed, not executed,
and not executed here). Causal model sections bound this plan: Y5 (contact identity and
delivery), Y7 (anatomy transfer), the qualification rules.

Executor: Vast 50616476 (`ssh braintrace-gpu`, `/workspace/braintrace-c3` worktree on branch
`campaign/h01-c3-partner-20260916`). Nothing in this campaign runs on the laptop except unit
tests and the Reasoning Studio exports. Register: the progressive search tree
(`docs/evidence/h01-reasoning/h01-search-tree.reasoning.json`, split Q-C3) and the campaign
prerequisite tree (`docs/evidence/h01-reasoning/h01-c3-campaign.reasoning.json`).

## Question

Does the non-proofread H01 C3 release supply human pyramidal cells that connect to the 17
kept cells, human interneurons that hold a test-to-failure gate at the deployed donor
densities, and anatomical synapses among them, so that Example 21's H01 backend starts from
a connected E/I core instead of 17 isolated cables?

## What the release provides (verified 2026-09-16, read-only)

| Product | Path | Content |
| --- | --- | --- |
| C3 cell table | `gs://h01-release/data/20210601/c3/segment_properties/info` | 46,637 segments with tags: layer L1-L6/WM, class (8,598 pyramidal, 4,373 interneuron, 1,500 atypical spiny, 190 spiny-stellate, 622 unclassified neuron), modifiers, counts NVx NSO NSI NSIe NSIi NDe NAx NSp NCi NAis NMy Sp |
| C3 skeletons | `.../20210601/c3/skeletons` | sharded precomputed skeletons with radius for every segment (kept cell 1684504313: 28,643 vertices, 8.6 mm cable, 24 components; proofread SWC: 63,526 nodes, 267 components) |
| C3 subcompartments | `.../20210601/c3/subcompartments` | 64 x 64 x 66 nm label volume: 100 axon, 101 dendrite, 102 astrocyte, 103 soma, 104 cilium, 105 AIS, 1100-1103 myelin; the proofread SWC type codes plus 100 (`braintrace/datasets/h01_anatomy.py` LABELS) |
| C3 synapse export | `.../20210729/c3/synapses/exported/` | 166 Avro shards, 32.9 GB, ~1M rows each; `pre_synaptic_site.neuron_id`, `post_synaptic_partner.neuron_id` (C3 ids), `type` 1 = I / 2 = E, location in 8/8/33 nm voxels, class labels, confidence |

Kept-cell C3 ids come from `docs/evidence/h01-cellmatrix-mapping-audit.json`
(`existing_spatial_owners` to `c3_ids`); seven of the 17 differ from the released id.

## What is held fixed

- No donor is imported, nothing is fitted, no density or anatomy is tuned. Every candidate is
  painted with the deployed donor profile for its type, unchanged, by the production builder
  (`make_h01_ei_cell`, `_regions`, MaxCVLen 10 um, implicit calcium solver, dt 0.005 ms).
  Y7 says this transfer fails on most 2-4 x 10^3 um2 components; this campaign measures how
  often it succeeds across 40 more anatomies. It does not repair the Y7 condition.
- The 25 dropped interneurons and the 62 dropped pyramidal cells stay dropped.
- The 17 kept cells keep their proofread SWCs; synapse endpoints are projected onto them.
- Manifest synapse kinetics stay as `H01Topology._validate` forces them (E 0 mV / 2 ms,
  I -80 mV / 5 ms, delay 0.5 ms).

## Steps and registered predictions

### A. C3 neuron synapse graph (box, `/workspace/venv-c3`)

`docs/evidence/h01_c3_cell_table.py` writes `docs/evidence/h01-c3-cell-table.json` from the
segment-properties file (sha-pinned copy under `.cache/h01/`). `docs/evidence/h01_c3_neuron_graph.py`
streams all 166 shards (download, scan, delete, atomic progress; reused from
`h01_full_export_join.py`) and keeps rows whose pre and post `neuron_id` are both tagged
`neuron`; output `docs/evidence/h01-c3-neuron-graph.json.gz` plus per-shard receipts.
`docs/evidence/h01_c3_partner_rank.py` ranks partners of the kept 17 by synapse count and
selects top 20 E (pyramidal, atypical spiny, spiny-stellate; layer tagged) and top 20 I
(interneuron; layer tagged); donors by `donor_for_tags`. Output
`docs/evidence/h01-c3-candidates.json` with the induced subgraph over kept and candidates.

Prediction A: every kept cell has at least one neuron partner in the graph (NSI 300-2,300
per kept cell); at least 40 partners with two or more synapses to the kept set exist.
Falsifier: fewer than 20 partners in total; then the pool is what exists and the campaign
reports the count.

### B. C3 skeletons as proofread-format SWC (box)

`docs/evidence/h01_c3_skeleton_export.py` fetches each candidate skeleton, splits it into
connected components, samples the subcompartment label at every vertex, and writes
`{id}.{k}.swc` in the proofread layout (x, y, z in 32/32/33 nm voxels, radius in nm, type =
label - 100, 0 -> -1). Archive `.cache/h01/c3-candidates-20260916.zip` with sha256 and
per-cell provenance (`non-proofread`). Production change: `H01Archive` accepts an explicit
expected digest; `H01ArchiveSet` resolves a source by `archive_sha256`. Validation through
the unchanged import path (`h01_population_import_audit.py --archive`), components file in the
`h01-population-components.json` shape.

Prediction B: 40 of 40 candidates import, construct and initialise. Control: kept cell
1684504313 exported from C3 and compared with its proofread SWC (recorded, not a gate).

### C. Test-to-failure on the 40 candidates and the 17 kept cells (box, two chains)

Runner `docs/evidence/h01_anatomy_transfer_run.py` gains `--ramp-na` (0 to 3x the donor's
test current over the pulse window) and reports rheobase and, where it occurs, the block
current; the step run and the dt-half repeat are unchanged. Chains from
`h01_keep_drop_chain.py`, follower and receipts unchanged.

The gate splits in two (J, 2026-09-16, while the chains ran; the raw measurements and the
runs are unchanged, only the verdict). (a) Hard failure edges, with datums from the human
recording and a tolerance of one sweep step of drive. (b) Plausibility rules, whose datum is
the donor's own value and whose tolerance is the across-cell spread of the human cells of the
donor's type in the Allen Cell Types database: 2 sd (sample sd across cells; the 5-95 percent
range is recorded beside it, not used). The single-donor bands (count band across the sweeps
within one step of the primary; 3 across-sweep sd of the donor's own pre-pulse and post-pulse
family levels) and the former bands (rest within 10 mV, count within max(2, 30 percent)) are
recorded beside every verdict for comparison and are not the gate.

| Edge | Model reading | Human datum | Tolerance | Pass |
| --- | ---: | --- | --- | --- |
| Hard: finite, no spike before the pulse | output and soma traces; -20 mV crossings before onset (step and ramp) | — | — | finite; zero crossings |
| Hard: firing range | rheobase from the ramp; last spike current on the ramp; block current | lowest firing and highest still-firing long-square amplitude (`long-square-index.json` / NWB stimulus, LJP-corrected) | one human sweep step | rheobase within the step; still fires at the human's highest amplitude; no depolarisation block at or below it (`no_block_in_recorded_range`); the block edge above it is recorded, not scored |
| Hard: numerical | dt-half repeat | primary run | exact count | reproduces |
| Plausibility: count | step count at the primary input | the donor's own count at the primary (`measured_counts_at_primary` mean: L2 10, L4 12, PV 12, SST 13.25) | 2 across-cell sd of `num_spikes` on the type population's 1 s long-square sweeps within one step of the primary drive (null = 0 spikes; `type_population.count.tolerance`: spiny L2/3 at 310 pA 14.2 over 147 cells, spiny L4 at 90 pA 20.1 over 37, aspiny at 190 pA 62.2 over 67, aspiny at 100 pA 43.1 over 77) | inside |
| Plausibility: pre-pulse rest | mean over the 100 ms before the pulse | `rest_repeat_mean_mv` (L2 -84.01, L4 -80.60, PV -87.12, SST -77.50 mV) | 2 across-cell sd of ipfx `pre_vm_mv` (500 ms before onset) on the same matched sweeps, -14 mV LJP: L2/3 7.7, L4 8.3, aspiny 9.4 / 9.8 mV | inside |
| Plausibility: post-pulse level | mean over the 10 ms ending 200 ms after the pulse (`h01_anatomy_transfer_run.windows`) | `after_repeat_mean_mv` (L2 -84.94, L4 -81.03, PV -87.18, SST -79.18 mV; every sweep of all four families reaches the window) | 2 across-cell sd of ipfx `post_vm_mv` on the matched sweeps: L2/3 7.6, L4 8.2, aspiny 9.3 / 9.7 mV. The Allen table has no field for the 10 ms ending 200 ms after offset; `post_vm_mv` is the mean over the last 500 ms of the recording (5-7 s after offset), the closest documented post-stimulus level, and is named as the fallback | inside; a trace that does not reach the window fails |

Type populations (`h01_allen_type_population.py`, sha-pinned pulls `.cache/h01/allen-human-ephys-features.json`
b19ed0b5... and `allen-human-ephys-sweeps.json` b3bd3d51..., `human-datums.json` `type_population`
/ `allen_population`): human specimens by `tag__dendrite_type` and `structure__layer` — spiny
layer 2/3 for the L2 donor (199 cells), spiny layer 4 for the L4 donor (37), aspiny of every
layer for both interneuron donors (79): the API exposes no fast-spiking / non-fast-spiking
label for human cells, so the split J named is not available and the aspiny population is the
closest label. Both single-donor count bands (L2 290/310/330 pA -> [9, 12]; L4 90/110 -> [12, 17];
PV 170/190/210 -> [4, 23]; SST 90/100 x4/110 -> [12, 21]) and both single-donor rest bands
(3 across-sweep sd: pre-pulse 0.92-1.42 mV, post-pulse 2.87 / 5.49 / 1.34 / 6.82 mV) stay in the
receipts as `donor_band_rules`.

The block edge above the recorded range is a model measurement and is recorded, not scored;
a block edge at or below the human's highest recorded amplitude is a hard failure. Usable-tier features (`h01_usable_tier.py`) are recorded on every cell. The 17 kept cells get
the same ramp; a kept cell failing the firing-range gate is reported to J, not dropped.

Prediction C-E: 1-4 of 20 pyramidal candidates pass (the former bands alone would pass 3-6).
Prediction C-I: 0-2 of 20 interneuron candidates pass (Y7). Prediction C-17: about half of the
13 kept L2 cells show a block edge under 3x input; the L4 cells hold. Falsifier for the
interneuron route: 0 of 20 -> a fit-pilot decision request is written for J; no tolerance is
widened and no cell is re-run.

### D. Contacts, manifest, bounded launch (box)

`h01_connectivity.select_connectivity/prepare_connectivity` accept the C3 graph: identity is
the table's C3 `neuron_id` (the segmentation the skeletons come from), so the proofread
"exact endpoints not verified" voxel gate does not apply; placement is the recorded
projection distance onto the cable (blocker beyond 3 um on the proofread 17, expected ~0 um on
candidates). Delivery is a weight sweep per anatomical contact at the post soma: the
unresolvable edge (PSP under 3 sd of the post cell's rest noise) and the fires-alone edge;
datum the literature pin (`h01-ie-synapse-literature.json`, 3.1 nS, 0.5-1 mV). I->E contacts
are swept with the post cell held depolarised (reversal -80 mV above the -84 mV rest). The
manifest weight must lie inside the resolvable range. Then `examples/h01_arc_probe.py` and
`examples/h01_arc_manifest.py` with two archives, and one launch of
`21-braincell-arc.py evolve --model-backend h01 --rounds 1 --patience 1 --screen-tasks 2`
under `timeout 3600`, receipted.

Prediction D: at least one construction-ready contact among the passing cells. Falsifier:
none -> the campaign stops at the network JSON and reports.

## Caps

Shard scan ~25 min wall (6 workers, to be measured on the first shard). Step C: 57 primary
runs + 57 ramps + repeats, two chains, cap 1800 s per step run, 3600 s per repeat; the first
chain's timing is recorded before the second is queued. Step D launch: 3600 s. Total box
time is measured and reported, never quoted in advance.

## Closing

Each step closes with its decision JSON, the search-tree node status, and the causal-model
update (Y5 for contacts, Y7 for the population extension) in one commit. Fast-forward into
`main` only on J's instruction.
