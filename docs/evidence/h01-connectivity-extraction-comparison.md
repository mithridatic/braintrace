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
