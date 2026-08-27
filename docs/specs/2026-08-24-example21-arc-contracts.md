# Example 21 ARC data and artifact contracts

## Scope

This specification covers OpenSpec tasks 2.1 through 3.5 for the BrainCell
ARC replacement. It defines the public contracts for raw data, temporal
events, request loss, direct decoding, strict results, and checkpoints.

## Implementation API

The authority module `examples/pp_prop/21-braincell-arc.py` exposes the
co-located contract implementation from `examples/pp_prop/arc_contracts.py`.
It provides
`load_task`, `encode_episode`, `decode_episode`, `request_loss`,
`decode_prediction`, `query_exact`, `strict_task_pass_at_1`, `write_result`,
`write_checkpoint`, and `load_checkpoint`. These functions use NumPy arrays
and immutable dataclasses at the boundary. A loader call names the role;
the default role is `practice`, and evaluation data is rejected unless the
caller explicitly passes `allow_evaluation=True`.

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
invalid-row steps have zero loss. The direct decoder consumes 31 request-state
vectors: one shape request followed by thirty row requests. It selects height
and width independently from 30 logits each, then selects one color from ten
logits for each cell of the selected rectangle. It returns a rectangular `uint8` grid
and consumes no target or query residual. A 360-value compatibility readout
remains accepted for old fixture checks and repeats one color across each
decoded row.

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
labels, and connection limits. It returns arrays only; the caller owns any
biological or eligibility reset. Writes use a temporary sibling and atomic replacement. A child path must
differ from its parent, and child failure must leave parent bytes unchanged.

The required array schema is fixed. It contains `neuron_ids` (`int32`, shape
`[N]`), `dale_codes` (`int8`, `[N]`), `owner_codes` (`int16`, `[N]`),
`mechanism_codes` (`uint8`, `[N]`), `neuron_count` and
`integration_substeps` (scalar `int32`), input and recurrent
`*_indptr`/`*_indices` (`int32`) and `*_values`/`*_m1`/`*_m2` (`float32`),
`readout_weight`/`readout_bias` and their `*_m1`/`*_m2` arrays (`float32`),
and scalar `input_step`, `recurrent_step`, and `readout_step` (`int64`).
The input CSR has shape `(1, neuron_count)`: `input_indptr` has shape `(2,)`
and input indices are neuron destinations. The recurrent CSR has shape
`(neuron_count, neuron_count)`, with `recurrent_indptr` shape
`(neuron_count + 1,)`; its row indices are source neurons and its column
indices are neuron destinations. The direct readout has weight shape
`(neuron_count, 360)` and bias shape `(360,)`, with moments matching those
shapes. CSR arrays must agree with their neuron count and endpoint bounds. Dale values
are limited to `-1`, `0`, and `1`; owner values are `-1`, `-2`, or nonnegative;
mechanism codes are nonnegative; all moments and parameters are finite; and
each CSR value/moment triplet has matching length. No extra arrays are
accepted. The total biological CSR entries are limited to 30,496.
