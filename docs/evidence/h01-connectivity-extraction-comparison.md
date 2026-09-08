# Connectivity extraction comparison (2026-09-08)

Read-only comparison; no new query, download campaign, simulation, or model edge.

The authors' get_edge_list_for_pr_cells_and_unproofread_targets.py maps the
proofreading records' base segments through agg20200916c3_resolved_fixed,
queries synaptic_connections_ei_conserv_reorient_fix_ei_merge_correction2 by
presynaptic axon base IDs, and groups results by postsynaptic agglomeration ID.
It writes both an edge list and a file-to-agglomeration mapping. Its default
includes unproofread targets; that output is not a closed 104-by-104 network.
Source: https://github.com/ashapsoncoe/h01/blob/main/get_edge_list_for_pr_cells_and_unproofread_targets.py

Our docs/evidence/h01_c3_edge_list.py instead samples skeleton positions into
the 20210601 C3 volume, queries annotation relationships for sampled labels,
and requires endpoint agreement against the separate proofread volume. This
is not a reproduction of the authors' base-segment/database join. It can miss
unsampled segments, and background endpoint reads remain unresolved rather
than evidence that the published connection is absent.

Next actionable dependency: locate the authors' released edge list AND its
file2aggloid mapping, or obtain the version-matched mapping/table access used
by the script. Check release identity and distinguish internal 104-cell edges
from outgoing edges to cells outside the selected set. Do not increase spatial
sampling again as a substitute for this join. Verify synapse coordinates and
cable placement separately from whether a published connectivity edge exists.

The accepted two-contact simulation remains a restricted implementation.
Neither the size of the recoverable internal graph nor a mapping defect causing
all rejected contacts has been established. No physiological conclusion changes.

## Public artifact lookup

Read-only listings on 2026-09-08 found no generated 104-cell edge list or
file2aggloid map in the authors' recursive GitHub tree or the paper bucket root.
The graph script also points to a local C:/work/FINAL edge-list file, rather
than a download URL. This is a bounded lookup, not proof of global absence.
The repository README says some BigQuery databases require private credentials;
we have not established access to the exact tables named by the extractor.

The release proofread_104/synapse_locations.csv is already imported locally:
264090 rows, SHA256 640ccb12c75b930f96373c7273bbf9688aa111b97b425ee22d8ef6db19f496be.
It supplies one proofread-cell ID and its pre/post role per row. The existing
h01-partner-coordinate-audit.json found zero candidate cell pairs under its exact
endpoint-coordinate join. This is not a newly discovered source or a complete
partner table. No redundant full download was performed.

The public c3/tables/segments directory lists 452 counts*.csv.gz objects. A
streamed header check found c3_id and compartment/synapse counts, not base IDs
or base-to-agglomeration membership. Those count tables cannot supply the needed
join. The base directory listing contains volume scales and mesh, with no
mapping table at that level. No bulk table download was performed.

Next remaining routes are supplementary analysis artifacts or a release-matched
base-to-agglomeration source. Recoverability and the number of internal 104-cell
edges remain unproven. Do not replace this gap with another spatial-sampling
campaign or silently promote the 126 spatial candidates.

## Supplementary cell table candidate

cellmatrix_404_c3.mat (1142209 bytes), SHA256
d4203c00c1e1995273b1e96543a3e51f84fe5de44ce26349d41cfbac3a608839,
was downloaded to .cache/h01. H01_multisynaptic_stats.m documents column 5 as
google agglomerated segment ID and column 6 as google base segment ID.
Direct numerical overlap with H01 IDs is only 53 in column 5 and 7 in column 6;
that overlap alone is not an identity mapping.

A membership join using every proofreading record's base_segments found exactly
one table base ID in each of all 104 records. This makes the table a candidate
104-cell mapping source. Before promotion, check duplicate table base IDs,
uniqueness of mapped C3 IDs, and release identity against embedded cb_agglo_lookup
and existing spatial evidence. The initial exploratory dictionary join did not
check duplicate table keys, so this is not yet an accepted mapping.

The archive also contains cb_agglo_lookup keyed by a named agglomeration release;
its first record includes main_agglo_seg 751294744. Inspect those embedded
mappings before performing more spatial queries. No model connectivity changed.

The saved h01-cellmatrix-mapping-audit.json now confirms one matched table
base ID per record, no duplicate matched base keys, and 104 distinct C3 IDs.
Only 95 match the archive main_agglo_seg in agg20200916_resolved; nine differ.
The archive release key is not the authors script
agg20200916c3_resolved_fixed. Preserve those nine explicit mismatches and resolve
release semantics before accepting the table as the final identity bridge.

All 104 candidate C3 IDs occur in the already saved 3000-sample spatial
lookup, each owned by exactly one distinct H01 cell. This establishes a
104-record association with the existing lookup without new spatial requests.
The embedded comparison consists of 95 agreements, seven differing IDs and
two absent entries, rather than nine conflicting IDs. The table association
does not map every axon base segment or itself supply synaptic connections.
The audit now retains each associated H01 cell and the spatial checkpoint hash.
