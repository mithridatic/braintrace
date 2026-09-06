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

## Outcome

The offline graph and contact placement export are complete, and 49 focused
tests pass with 100% line coverage of the two new library modules. All three
verified contacts remain in the anatomical graph. Two pass cable-placement
gates; one starts on a soma-free fragment. The full real four-cell construction
attempt did not finish within about eight minutes and was stopped. Full-scale
construction/runtime remain unverified; the delivered builder is fixture-tested.
See docs/evidence/h01-verified-network-validation.md for the exact boundary.
