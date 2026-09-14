# H01 human-reference topographic search tree

Status: implemented 2026-09-09; structural/evidence checks passed. Native
application open/save/reopen and visual inspection remain unverified; see the
verification record in `docs/diagnosis/` for the application-access boundary.

## Purpose

Create a current-diagnosis Flying Logic document, not a reconstruction of every
campaign. Each observed discrepancy has its own binary search subtree. Cover
reference/measurement, E and I donor responses, simulator/anatomy transfer,
contact/circuit function, and population construction and reference coverage.
The overview organizes these concurrent problems; its links are navigation,
not a mutually exclusive causal partition.

## Artifacts

Under `docs/diagnosis/`, provide `h01-human-reference-search.xlogic`, a matching
Markdown companion, a JSON record of questions and branches, and a verification
record. Build native Flying Logic 5 XML from the installed Flowchart example's
serialization conventions. Use left-to-right layout, explicit Yes/No labels,
stable IDs, short titles, detailed annotations, and hidden confidence/weights.
No model API or simulation code changes are needed.

## Decision semantics

Every question has two complementary outcomes over its stated remaining space.
Node status is answered Yes, answered No, open, inconclusive, or blocked.
Missing data and invalid tests are not No. Evidence of a failed intervention
excludes only that intervention under the tested conditions. Interactions and
untested mechanisms remain possible. A predicted failure is not an observation.

Each annotation retains the response coordinates, scope and version, evidence,
decision limit, Yes/No interpretation, next check, and unresolved boundaries.
Use hierarchical characterization before functional isolation, structural
dissection, or energetic splits. Voltage and current/charge remain paired at
energetic boundaries. No inference of an exhaustive mechanism list is allowed.

## Qualification and evidence

Preserve the approved exact counts, 1 ms timing, 1 mV voltage, and 0.05 ms
spike-phase contract per event/input. Usable-tier, numerical, transfer, runtime,
anatomical, and physiological verdicts remain separate. H01 lacks matching
voltage recordings; donor-constrained H01 physiology is a prediction.

Evidence snapshot: base commit 34ca482. Prefer dated machine-readable decisions
over older summaries. Retain candidate IDs, stimulus and mesh differences,
measurement windows, opened holdouts, and stopped/capped campaigns. Cite book
OCR files pNNN.md, with printed page numbers only as secondary labels. Use
topographic throughout. Physical laws, engineering limits, and assumptions are
different kinds of statements.

## Verification

Check XML well-formedness, unique IDs, class references, valid graph endpoints,
acyclic structure, two labeled outcomes per decision, no repeated consecutive
predicate, evidence existence, JSON/Markdown/XML agreement, and evidence hashes.
Check that unresolved states never masquerade as No, unrun interventions are
not labeled observed failures, and cell/network construction does not imply
human validation. Open, save, reopen, and inspect in Flying Logic when available;
otherwise report the exact unverified application boundary.

## Execution boundary

Work only on `docs/h01-flying-logic-search-tree` in an isolated worktree. This
artifact task does not run simulations, fit models, open holdouts, repair
morphology, or revive stopped campaigns. Leaves specify future discriminating
checks; carrying them out requires a separately bounded experiment plan.
