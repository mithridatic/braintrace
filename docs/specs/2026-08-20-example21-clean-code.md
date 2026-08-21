# Example 21 clean-code training worker reduction

Status: implemented
Date: 2026-08-20

## Goal

Reduce Example 21 source size without changing the live training, model,
optimizer, evaluation, or scoring paths.

## Finding

The scalar row worker path (`_materialize_training_row` and
`_ordered_training_rows`) is no longer called by the production training
pipeline. The current pipeline materializes deterministic episode descriptors
with `_materialize_training_episode` and consumes them through
`_ordered_training_episodes`, including when the training batch size is one.
The scalar path is retained only by obsolete tests.

## Change contract

- Remove the unreachable scalar worker implementation and its obsolete tests.
- Keep the live episode worker's serial and bounded parallel behavior unchanged.
- Do not change random seeds, task selection, encoded tensors, optimizer inputs,
  model parameters, evaluation checkpoints, candidate selection, or metrics.
- Do not add a compatibility shim for a private path with no production caller.

## Acceptance

- `rg` shows no production reference to the removed scalar worker symbols.
- The co-located Example 21 tests for training workers, ordering, cleanup,
  serial/parallel equivalence, and batched/scalar tensor equivalence pass.
- The complete co-located Example 21 test module passes.
- Production source line count is lower than the base commit.
- A deterministic smoke run produces the same reported metrics and candidate
  outputs as the base commit, and the cleanup does not add runtime work.

## Verification

- Production entry-point source: 5,962 lines at `324d903` → 5,900 lines
  (`-62`). Its co-located test: 4,405 → 4,370 lines (`-35`).
- Focused worker/training gate: 6 passed.
- Complete co-located Example 21 gate: 150 passed, 20 existing compiler
  warnings.
- Identical smoke configuration, seed 2108, CPU, 128 neurons, 1,024 edges,
  three updates, and 60 latent steps: metrics, checkpoint candidates, losses,
  and parameter-change evidence compared equal between `main` and this branch.
- Recorded smoke runtime: 22.165 seconds on `main`; 19.186 seconds on this
  branch. Wall time was 29.764 seconds and 27.753 seconds respectively.
- Ruff 0.16.4 reports the same 12 unique diagnostic signatures on both
  revisions, with zero added or removed by this cleanup; the direct check
  remains non-zero because those diagnostics predate this branch.
- BasedPyright remains a broad baseline failure on these files (351 errors and
  4,430 warnings); the cleanup does not touch any reported diagnostic site.
- This is a smoke equivalence/performance result, not full ARC scientific
  qualification evidence.
