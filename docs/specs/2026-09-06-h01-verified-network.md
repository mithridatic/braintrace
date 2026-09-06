# Assemble the verified H01 connections

Status: implementation authorized by the user's instruction to set up the
verified connections, following the proposed identify, place, and add workflow.
Work is confined to feat/h01-braincell. No physiology fitting is part of this task.

## Deliverable

Provide an offline connection table with all 104 cell IDs, source tags,
E/I assignments, and Dale signs. Select only contacts whose exact endpoints
are verified in h01-resolved-edge-list.json. The current report contains three
such contacts (EE, EI, IE) across six cells, out of 123 candidates. Preserve rejected
candidate IDs and reasons; missing edges do not establish biological absence.

Construct a BrainCell network for the cells incident on accepted contacts.
Keep all 104 identities in the topology export, including cells without an
accepted connection. Do not present those other cells as simulated populations.
Reuse the existing candidate electrical profiles, source-label region policy,
and synapse mechanism. Do not introduce the diagnostic sodium override or
promote the newer potassium finalist. Support disconnected controls.

## Contact and sign rules

Verify the source archive checksum, endpoint labels, zero-offset checks, and
agreement between presynaptic cell type and synapse type. Deduplicate by
annotation ID and reject conflicts. Pyramidal and excitatory atypical cells
map to E/+1; interneurons map to I/-1. Retain original source tags.

Locate each endpoint on the source cable, recording component, location,
distance, and source hash. Accept only unambiguous cable placements within
1 micrometer. Never bridge disconnected pieces or relocate a contact to a
soma. A contact without a supported soma-containing component remains in
the verified anatomical table with an explicit construction blocker.

Use nonnegative conductance magnitudes. Select excitatory or inhibitory
receptor parameters from the sending cell's sign, regardless of receiver type.
Export signed adjacency separately: sign times conductance magnitude.
Default assumed conductances are E 0.01 uS and I 0.02 uS, delay 0.5 ms,
reversals 0/-80 mV, and decay constants 2/5 ms, as in the existing circuit.
These are model assumptions, not measured synapse properties.

## Verification and cost boundary

Co-located tests cover selection, signs, unknown labels, duplicate/conflicting
contacts, invalid parameters, rejected placements, isolated cells, and
connected/disconnected construction. Use small fixtures for delivery checks.
Run the offline export and construct the real accepted network. A short
compiled smoke run may check finite states; no long waveform fitting or
full physiological qualification is required. Report construction and actual
simulation separately. Example 21 training integration is a later task.

## Correction

Earlier status messages reported only the original I-to-E implementation and
missed the newer population edge audit. Read the canonical full edge report
when describing connection inventory; do not infer it from a two-cell demo.

## Initial outcome (superseded by resumed construction)

The offline graph and contact placement export are complete, and 49 focused
tests pass with 100% line coverage of the two new library modules. All three
verified contacts remain in the anatomical graph. Two pass cable-placement
gates; one starts on a soma-free fragment. The full real four-cell construction
attempt did not finish within about eight minutes and was stopped. Full-scale
construction/runtime remain unverified; the delivered builder is fixture-tested.
See docs/evidence/h01-verified-network-validation.md for the exact boundary.

## Resumed construction

The user requested completion of the real construction on 2026-09-06.
Add elapsed-time progress at source loading, location validation, electrical
cell construction, and discretization/population registration. Preserve the
same cell models, geometry, contacts, and numerical settings. Do not stop merely
because the previous eight-minute limit has elapsed. If a construction defect
is found, reproduce it in a co-located test before fixing it. Run a short
compiled finite-state check after construction if resources permit, keeping
that check separate from physiology and synaptic-response validation.

### Branch-index construction defect

The resumed run loads and checks all four shapes in 42.5 seconds, then spends
minutes constructing the first electrical cell. Installed BrainCell geometry
and node-tree builders query MorphoBranch.index per edge. Each query rebuilds
the entire default-order index map in Morphology._branch_index_map.

Use an H01 Cell subclass whose discretization property scopes a default-order
index cache to the synchronous build on that morphology. Restore the original
lookup on exit, including exceptions; do not patch BrainCell globally or edit
the installed dependency. Other branch orders use the original lookup. Within
the scope, invalidate the map when node count or next allocated ID changes.
Geometry, mesh policy, current laws, source hashes, and model parameters must
remain identical. Validate index-work counts, geometry/axial-array parity,
exception cleanup, and cloned-morphology initialization before rerunning the
real construction. Cache only integer lookup results, not physiological states.

The first cache fixes initial cell construction. Final channel assignment has
the same quadratic pattern in _coverage_fraction: morpho.branches rebuilds the
whole ordered branch tuple per compartment. Extend the same temporary scope
to branch_by_order(default), retaining the original branch views and restoring
both instance methods on exit. A separate reproducer observes 44 tuple builds
on a three-branch fixture; require at most two. Other orderings stay uncached.

## Resumed outcome

Full construction passes: four real H01 cells, two projections, and 75,605
compartments in 331.20 seconds. The blocked fragment contact remains in the
anatomical graph. Sixty focused tests pass with 100% line coverage for the
construction and network modules. The completed build evidence is saved in
docs/evidence/h01-verified-network-build.json. No physiological settings changed.

The subsequent one-step execution check rebuilt the circuit in 211.1 seconds
but failed before stepping: BrainCell's dense axial-operator reduction raised
MemoryError during initialization. Construction is verified; real execution
remains blocked by solver memory. The failure is preserved in
docs/evidence/h01-network-execution.log and the validation page.
