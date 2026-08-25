# Example 21 ARC data and artifact contracts

## Scope

This specification covers OpenSpec tasks 2.1 through 3.5 for the BrainCell
ARC replacement. It defines the public contracts for raw data, temporal
events, request loss, direct decoding, strict results, and checkpoints.

## Data

The loader reads `<arc-root>/data/training/<task-id>.json` for practice data
and `<arc-root>/data/evaluation/<task-id>.json` only when explicitly allowed.
Ordinary training and validation reject evaluation-role data. It reads JSON
directly and does not build an index, fingerprint, hash, or synthetic task.

Each grid is a non-empty rectangular `uint8` array with dimensions 1 through
30 and cell values 0 through 9. A task has at most ten demonstration pairs
and one or more test queries. Invalid grids and excess demonstrations fail
with a corrective error containing the task and observed value.

The fixed practice order is `d631b094`, `dc433765`, `b782dc8a`, `d06dbe63`,
`aedd82e4`, `0b148d64`, `b2862040`, `150deff5`. Validation is
`46f33fce`, `3428a4f5`, `d8c310e9`, `09629e4f`. Proof uses
`d631b094` and `46f33fce`.

## Events

An episode is a deterministic `(705, 441)` boolean event array plus a
705-element boolean advance mask. Features are sliced as event type 0:7,
role 7:11, demonstration slot 11:21, height 21:51, width 51:81, row
81:111, valid-cell mask 111:141, and position/color 141:441. Applicable
categorical fields are one-hot; unused fields are all zero.

The schedule contains the task start, ten fixed demonstration blocks, the
query input, input end, one shape request, and thirty row requests. Missing
rows and unused demonstrations are padded with false advance. The held-out
target is used only by loss and scoring, never by event encoding. Encoding
then decoding the inference portion reproduces all input grids exactly, and
changing only the target leaves inference bytes unchanged.

## Loss, prediction, and scoring

Shape loss is height cross-entropy plus width cross-entropy. Each row loss is
the mean cross-entropy over valid target cells. Non-request, empty, and
invalid-row steps have zero loss. The decoder selects height and width
independently from 30 logits each, then selects one color from ten logits for
each of 30 columns at each row request. It returns a rectangular `uint8`
grid and consumes no target or query residual.

Query exactness requires a non-Boolean integer prediction with equal shape and
equal cells. Strict task pass-at-1 is true only when every query is exact.
No partial, pixel, average, or tolerance score is produced.

The result root contains only `tasks` and
`strict_task_pass_at_1_count`. Task records contain only `task_id`, `queries`,
and `strict_pass_at_1`; query records contain only `query_index`,
`prediction`, `target`, and `exact`. JSON cells are integers and encoded size
is limited to 256 KiB. Writes validate a temporary sibling before atomic
replacement.

## Checkpoints

Checkpoints are compressed NumPy archives with `format == 1`, arrays only,
`allow_pickle=False`, and encoded size at most 32 MiB. They contain topology,
`float32` CSR values, readout parameters, all Adam moments, three step counts,
and neuron/substep counts. They do not contain targets, predictions, losses,
gradients, or runtime state. Load validates dtype, shape, endpoints, counts,
labels, and connection limits, then resets biological and eligibility state.
Writes use a temporary sibling and atomic replacement. A child path must
differ from its parent, and child failure must leave parent bytes unchanged.
