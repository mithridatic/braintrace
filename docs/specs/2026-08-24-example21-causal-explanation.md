# Example 21 causal explanation

## Scope

This document reports the executed direct-system evidence available in the
workspace. It does not report the planned BrainCell replacement as executed.

The acceptance measure is strict task pass-at-1. A task passes only when every
test query has the exact height, width, and cell values in the first model
prediction.

## Observations

- The fixed evaluation manifest contains 400 tasks and 419 test queries.
- The latest recorded direct artifact is
  `var/ex21-online-v48-query-routing-v1/result.json`.
- The artifact reports 12/120 synthetic holdout tasks, 0/53 real in-library
  queries grouped into 51 tasks, and 0/85 real fold-zero queries grouped into
  80 tasks. The fixed evaluation manifest was not run.
- The direct model uses two 128-unit MiniLSTM memories. It joins their states
  and their elementwise product into 384 values. It uses 12 color experts and
  16 routing programs.
- The output path produced one exact synthetic upscale result. It produced no
  strict real development task.
- A target-fed decoder check produced 400/400 strict tasks. This check used
  target-derived logits. It is a scorer check, not model performance.

## Inferences

The observations support a limited inference. The direct path can preserve
some task information and can produce exact answers for some synthetic tasks.
The main measured break is between task state and reliable exact-grid output.
This does not prove that every failed task was recognized correctly. It also
does not prove that PP-Prop equals BPTT.

The 400/400 target-fed result shows that the decoder and strict scorer can
represent the target grids. It does not show that the executed model produces
the required logits.

## Boundary

No executed BrainCell checkpoint, topology plot, or BrainCell prediction is
present in this worktree. Therefore this document makes no BrainCell neuron,
connection, Dale-type, or causal-success claim.
