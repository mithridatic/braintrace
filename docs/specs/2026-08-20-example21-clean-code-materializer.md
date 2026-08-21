# Example 21 remove the test-only whole-run materializer

Status: implemented
Date: 2026-08-20

## Objective

Reduce Example 21 production source without changing the live training,
optimizer, evaluation, scoring, or artifact paths. The cleanup removes the
whole-run `_prepare_training` materializer, which is not called by production,
and keeps its schedule-equivalence coverage in the co-located test module.

## Finding

`run_experiment` supplies the streaming `_training_chunks` iterable directly
to `_train_model`. `_prepare_training` is referenced only by tests and its
concatenation helper and field-name constants have no other callers. Keeping
that path in the production module exposes an obsolete API and retains a
whole-run allocation pattern that the streaming design intentionally removed.

## Change contract

- Remove `_prepare_training`, `_concatenated_chunks`, and their private field
  constants from the production entry point.
- Move whole-schedule assembly needed by equivalence tests into test-local
  support code; do not restore whole-run materialization to production.
- Preserve the schedule, encoded tensors, metadata ordering, optimizer inputs,
  model parameters, evaluation checkpoints, candidate selection, and metrics.
- Preserve the existing chunked-training and episode-bank behavior.
- Do not change public command-line options or JSON schemas.

## Acceptance

- The removed production symbols have no source, test, or documentation
  references except historical specifications that are updated to describe
  the new ownership.
- The co-located Example 21 test module passes, including chunk equivalence,
  batched/scalar equivalence, bank reproducibility, and training-mask tests.
- Production source line count is lower than the base commit.
- A deterministic CPU smoke run has equal metrics, candidates, losses, and
  parameter-change evidence against `main`, and is no slower than the base
  run within the measured repeat range.
- The cleanup adds no runtime work to the live path.

## Verification

Run the focused materializer/chunking tests, the complete co-located Example
21 test module, changed-file Ruff, and the deterministic smoke comparison.
Record edge cases covered by the test-local materializer: one chunk, multiple
chunks, different batch sizes, banked and fresh schedules, metadata ordering,
and zero-update structural construction.

## Verification

- Production entry-point source: 6,245 lines at `e2ab87b` -> 6,201 lines
  (`-44`). The co-located test grew from 4,945 to 4,980 lines (`+35`) to own
  the test-only oracle; the combined pair is still 9 lines shorter.
- Focused chunking, horizon, batching, masking, and bank checks: 8 passed.
- Complete co-located Example 21 gate: 149 passed, 20 existing compiler
  warnings.
- Python compilation, `git diff --check`, and module Ruff passed.
- Two independent smoke runs on each revision produced equal metrics,
  checkpoint candidates, trajectories, losses, effort schedules, parameter
  changes/hashes, qualification, and data manifests. Standalone CPU wall time
  varied with JAX compilation and host load; the same-process warm comparison
  measured base/candidate pairs of 28.865/12.999 seconds and 12.328/8.393
  seconds. The live training and evaluation path is unchanged by the diff.
